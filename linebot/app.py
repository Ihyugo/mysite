from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import  InvalidSignatureError
from linebot.models import  MessageEvent, TextMessage, TextSendMessage,

app = Flask(__name__)

line_bot_api = LineBotApi('cRXWc37IvbconSu/4F8cnTzDuRstUiPE5sD+Yz7JR4L2fdF3oZEM9c3ajfkJmFTT1YsFA9MFr6w3DQbUfqTTcm/MTI6OstuWpGG0xg92ToDtBsnFNZvQRRM4R5Ivj9B8MYu80fk60BK2CBv+6csD3QdB04t89/1O/w1cDnyilFU=')
handler = WebhookHandler('4dca934feb3fd9657112f7d3a9e0d319')


@app.route("/callback", methods=['POST'])
def callback():
    # get X-Line-Signature header value
    signature = request.headers['X-Line-Signature']

    # get request body as text
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)

    # handle webhook body
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)

    return 'OK'


@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=event.message.text))


if __name__ == "__main__":
    app.run()
