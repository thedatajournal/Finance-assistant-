import gradio as gr

from src.handlers import get_bot, respond
from src.utils import load_config


def main():
    server_cfg = load_config()["server"]
    get_bot()  # load models once at startup

    demo = gr.ChatInterface(
        fn=respond,
        type="messages",
        title="Finance Assistant",
        description="Ask finance and investment questions, answered using the latest news.",
    )
    demo.launch(server_name=server_cfg["host"], server_port=server_cfg["port"])


if __name__ == "__main__":
    main()
