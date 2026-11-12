"""这里是调试系统(简要)"""

class Debug_System:
    ## 调试选项
    is_debug: bool = True
    ## 调试(没什么分类)
    def debug_log(kind: str, message):
        """检查调试状态"""
        if not Debug_System.is_debug:
            return
        """调试"""
        source_kind: str = f"[ {kind} ]"
        debug_message: str = f"{source_kind}: {message}"
        print(debug_message)