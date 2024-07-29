import json, os
from pathlib import Path
from fastchat.protocol.openai_api_protocol import (
    ChatCompletionRequest,
)
from fastchat.constants import (
    ErrorCode,
)

from fastchat.serve.openai_api_server import get_worker_address, get_gen_params, create_error_response, fetch_remote

from fastchat.utils import build_logger
logger = build_logger("openai_api_server", "openai_api_server.log")

async def get_token_length(request, prompt, worker_addr):
    token_num = await fetch_remote(
        worker_addr + "/count_token",
        {"model": request.model, "prompt": prompt},
        "count",
    )
    return token_num


async def get_context_length(request, worker_addr):
    context_len = await fetch_remote(
        worker_addr + "/model_details", {"model": request.model}, "context_length"
    )
    return context_len

async def lcw_process(request: ChatCompletionRequest, worker_addr):
    MAX_NUM_MESSAGES = 12
    MAX_CONTEXT_LENGTH = 8000

    full_conv = "\n".join([json.dumps(m, ensure_ascii=False) for m in request.messages])
    messages = request.messages
    system_message = messages.pop(0)
    
    conv_file_path = None
    user_id = ""
    if request.user is not None and "|" in request.user:
        uu = request.user.split("|")
        if uu[0] == "vote":     # "vote|up/down|user_id|chat_id"
            newpath = f"{Path.home()}/log/saju-vote/{uu[1]}/{uu[2]}"
            if not os.path.exists(newpath):
                os.makedirs(newpath)
            file_name = f"{uu[3]}.jsonl"
            vote_file_path = os.path.join(newpath, file_name)
            with open(vote_file_path, "w") as f:
                f.write(full_conv)
            return "stop"
            
        user_id = uu[1]
        menu_title = ""
        if len(uu) > 2:
            menu_title = uu[2]
        newpath = f"{Path.home()}/log/saju-conv/{user_id}"
        if not os.path.exists(newpath):
            os.makedirs(newpath)

        file_name = f"{menu_title}_{uu[0]}.jsonl"
        conv_file_path = os.path.join(newpath, file_name)
        with open(conv_file_path, "w") as f:
            f.write(full_conv)
        logger.info(f"{user_id=}, {menu_title=}")    
    
    context_length = (await get_context_length(request, worker_addr)) - 128
    if context_length > MAX_CONTEXT_LENGTH:
        context_length = MAX_CONTEXT_LENGTH
    if len(messages) > MAX_NUM_MESSAGES:
        messages = messages[-MAX_NUM_MESSAGES:]
        
    # compact assistant message
    for idx in range(len(messages)):
        messages[idx]['content'] = messages[idx]['content'].strip()
        m = messages[idx]
        content = m['content'].strip()
        if m['role'] == 'assistant' and len(content) > 500:
            cc = content.split(".")
            i = 0
            begin = ""
            while len(begin) < 150 and i < len(cc):
                if len(cc[i]) > 0:
                    begin += cc[i] + "."
                i += 1
            i = len(cc) - 1
            end = ""
            while len(end) < 150 and i > 0:
                if len(cc[i]) > 0:
                    end = cc[i] + "." + end
                i -= 1
            messages[idx]['content'] = (begin + end).replace("\n\n", "\n")
            # messages[i]['content'] = content[:100] + "..." + content[-100:]
            # logger.info(f"compacted={messages[i]['content']}")
            
    # if len(messages) > 6:
    #     messages.insert(len(messages) - 1, {"role": "user", "content": "상기 대화 보다는 맨앞 운세 자료를 기반으로 아래 질문에 답변해."})

    messages.insert(0, system_message)
    # if "ChangGPT" not in system_message["content"] and "SajuGPT" not in system_message["content"]:
    #     messages[-1]["content"] = messages[-1]["content"].replace("사주", "운세[사주]")
        # messages[-1]["content"] += "(내 운명이 걸린 일이니 위 사주를 분석하여 답변하세요)"
        

    logger.info(f"{request.max_tokens=}")
    request.messages = messages
    max_tokens = request.max_tokens
    if not request.max_tokens:
        max_tokens = 500
    while True:
        gen_params = await get_gen_params(
            request.model,
            worker_addr,
            messages,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            presence_penalty=request.presence_penalty,
            frequency_penalty=request.frequency_penalty,
            max_tokens=max_tokens,
            echo=False,
            stop=request.stop,
        )
        input_length = await get_token_length(
            request, gen_params["prompt"], worker_addr
        )
        logger.info(f"{input_length=}\n{max_tokens=}\n{len(messages)=}")
        if input_length + max_tokens > context_length:
            if len(messages) == 2:
                return create_error_response(
                    ErrorCode.INTERNAL_ERROR, "message too long."
                )
            else:
                messages.pop(1)
        else:
            max_tokens = context_length - (input_length + 96)
            if max_tokens < 100:
                max_tokens = 100
            elif max_tokens > 2000:
                max_tokens = 2000
            if request.max_tokens and max_tokens > request.max_tokens:
                max_tokens = request.max_tokens
            break

    request.messages = messages
    # request.max_tokens = max_tokens
    gen_params = await get_gen_params(
        request.model,
        worker_addr,
        messages,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        presence_penalty=request.presence_penalty,
        frequency_penalty=request.frequency_penalty,
        max_tokens=max_tokens,
        echo=False,
        stop=request.stop,
    )

    # print(messages)
    print(gen_params["prompt"])
    logger.info(f"after calc {max_tokens=} {input_length=} {context_length=}")
    logger.info(f"max_new_tokens={gen_params['max_new_tokens']}")
    logger.info(f"{request.temperature=}, {request.top_p=}")


    return conv_file_path         


def extract_last_user_message(text):
    start_tag = "<|start_header_id|>user<|end_header_id|>"
    end_tag = "<|eot_id|>"
    
    last_start = text.rfind(start_tag)
    if last_start == -1:
        return None  # Start tag not found
    
    message_start = last_start + len(start_tag)
    message_end = text.find(end_tag, message_start)
    
    if message_end == -1:
        return None  # End tag not found
    
    return text[message_start:message_end]
