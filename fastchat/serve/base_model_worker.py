import asyncio
import threading
import time
from typing import List

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
import requests

from fastchat.constants import WORKER_HEART_BEAT_INTERVAL
from fastchat.conversation import Conversation
from fastchat.utils import pretty_print_semaphore, build_logger

worker = None
logger = None

app = FastAPI()


def heart_beat_worker(obj):
    while True:
        obj.idle = True # lcw

        time.sleep(WORKER_HEART_BEAT_INTERVAL)

        # if obj.idle and obj.get_queue_length() > 0:
        #     logger.info("worker idle. reset semaphore")
        #     obj.semaphore = asyncio.Semaphore(worker.limit_worker_concurrency)
            
        # if obj.get_queue_length() == 0 and not obj.auto_register:
        #     logger.info("now exit worker")
        #     import os
        #     os._exit(0)
            
        obj.idle = True
        
        obj.send_heart_beat()


class BaseModelWorker:
    def __init__(
        self,
        controller_addr: str,
        worker_addr: str,
        worker_id: str,
        model_path: str,
        model_names: List[str],
        limit_worker_concurrency: int,
        conv_template: str = None,
        multimodal: bool = False,
    ):
        global logger, worker

        self.controller_addr = controller_addr
        self.worker_addr = worker_addr
        self.worker_id = worker_id
        if model_path.endswith("/"):
            model_path = model_path[:-1]
        self.model_names = model_names or [model_path.split("/")[-1]]
        self.limit_worker_concurrency = limit_worker_concurrency
        self.conv = self.make_conv_template(conv_template, model_path)
        self.conv.sep_style = int(self.conv.sep_style)
        self.multimodal = multimodal
        self.tokenizer = None
        self.context_len = None
        self.call_ct = 0
        self.semaphore = None
        
        # lcw
        self.auto_register = True
        self.idle = True    

        self.heart_beat_thread = None

        if logger is None:
            logger = build_logger("model_worker", f"model_worker_{self.worker_id}.log")
        if worker is None:
            worker = self
        
    def make_conv_template(
        self,
        conv_template: str = None,
        model_path: str = None,
    ) -> Conversation:
        """
        can be overrided to costomize the conversation template for different model workers.
        """
        from fastchat.conversation import get_conv_template
        from fastchat.model.model_adapter import get_conversation_template

        if conv_template:
            conv = get_conv_template(conv_template)
        else:
            conv = get_conversation_template(model_path)
        return conv

    def init_heart_beat(self):
        self.register_to_controller()
        self.heart_beat_thread = threading.Thread(
            target=heart_beat_worker,
            args=(self,),
            daemon=True,
        )
        self.heart_beat_thread.start()

    def register_to_controller(self):
        logger.info("Register to controller")

        url = self.controller_addr + "/register_worker"
        data = {
            "worker_name": self.worker_addr,
            "check_heart_beat": True,
            "worker_status": self.get_status(),
            "multimodal": self.multimodal,
        }
        r = requests.post(url, json=data)
        assert r.status_code == 200

    def send_heart_beat(self):
        logger.info(
            f"Send heart beat. Models: {self.model_names}. "
            f"Semaphore: {pretty_print_semaphore(self.semaphore)}. "
            f"call_ct: {self.call_ct}. "
            f"worker_id: {self.worker_id}. "
        )

        url = self.controller_addr + "/receive_heart_beat"

        while True:
            try:
                data = {
                    "worker_name": self.worker_addr,
                    "queue_length": self.get_queue_length(),
                }
                ret = requests.post(
                    url,
                    json=data,
                    timeout=5,
                )
                exist = ret.json()["exist"]
                logger.info(f"heart beat: {data} worker exist: {exist}")
                break
            except (requests.exceptions.RequestException, KeyError) as e:
                logger.error(f"heart beat error: {e}")
            time.sleep(5)

        if not exist and self.auto_register:
            self.register_to_controller()

    def get_queue_length(self):
        if self.semaphore is None:
            return 0
        else:
            sempahore_value = (
                self.semaphore._value
                if self.semaphore._value is not None
                else self.limit_worker_concurrency
            )
            waiter_count = (
                0 if self.semaphore._waiters is None else len(self.semaphore._waiters)
            )
            logger.info("semaphore value: %d, waiter count: %d", sempahore_value, waiter_count)
            return self.limit_worker_concurrency - sempahore_value + waiter_count

    def get_status(self):
        return {
            "model_names": self.model_names,
            "speed": 1,
            "queue_length": self.get_queue_length(),
        }

    def count_token(self, params):
        prompt = params["prompt"]

        try:
            input_ids = self.tokenizer(prompt).input_ids
            input_echo_len = len(input_ids)
        except TypeError:
            input_echo_len = self.tokenizer.num_tokens(prompt)

        ret = {
            "count": input_echo_len,
            "error_code": 0,
        }
        return ret

    def get_conv_template(self):
        return {"conv": self.conv}

    def generate_stream_gate(self, params):
        raise NotImplementedError

    def generate_gate(self, params):
        raise NotImplementedError

    def get_embeddings(self, params):
        raise NotImplementedError


def release_worker_semaphore():
    worker.semaphore.release()


def acquire_worker_semaphore():
    if worker.semaphore is None:
        worker.semaphore = asyncio.Semaphore(worker.limit_worker_concurrency)
    worker.idle = False
    return worker.semaphore.acquire()


def create_background_tasks():
    background_tasks = BackgroundTasks()
    background_tasks.add_task(release_worker_semaphore)
    background_tasks.add_task(worker.send_heart_beat)  # lcw
    return background_tasks


@app.post("/worker_generate_stream")
async def api_generate_stream(request: Request):
    params = await request.json()
    await acquire_worker_semaphore()
    generator = worker.generate_stream_gate(params)
    background_tasks = create_background_tasks()
    return StreamingResponse(generator, background=background_tasks)


@app.post("/worker_generate")
async def api_generate(request: Request):
    params = await request.json()
    await acquire_worker_semaphore()
    output = await asyncio.to_thread(worker.generate_gate, params)
    release_worker_semaphore()
    worker.send_heart_beat()  # lcw
    return JSONResponse(output)


@app.post("/worker_get_embeddings")
async def api_get_embeddings(request: Request):
    params = await request.json()
    await acquire_worker_semaphore()
    embedding = worker.get_embeddings(params)
    release_worker_semaphore()
    return JSONResponse(content=embedding)


@app.post("/worker_get_status")
async def api_get_status(request: Request):
    return worker.get_status()


@app.post("/worker_stop_auto_register")
async def api_stop_auto_register(request: Request):
    worker.auto_register = False
    return {"status": "ok"}


@app.post("/count_token")
async def api_count_token(request: Request):
    params = await request.json()
    return worker.count_token(params)


@app.post("/worker_get_conv_template")
async def api_get_conv(request: Request):
    return worker.get_conv_template()


@app.post("/model_details")
async def api_model_details(request: Request):
    return {"context_length": worker.context_len}
