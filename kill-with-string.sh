ps aux | grep "$1" | grep -v grep $( [ -n "$2" ] && echo "| grep -v \"$2\"" ) | awk '{print $2}' | xargs kill
