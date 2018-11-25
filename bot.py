"""
Init boll and start polling
"""

import vk_api
from vk_api.bot_longpoll import VkBotLongPoll, VkBotEventType
from skimage import io
import numpy as np
import cv2
from nets import model
from text_extraction import meme_info

api_token = #your token
club_id = #your club id


def main():
    """ Пример использования bots longpoll
        https://vk.com/dev/bots_longpoll
    """

    vk_session = vk_api.VkApi(token=api_token)
    vk = vk_session.get_api()
    longpoll = VkBotLongPoll(vk_session, club_id)

    for event in longpoll.listen():
        if event.type == VkBotEventType.MESSAGE_NEW:
            if event.obj.attachments:
                attach = event.obj.attachments[0]
                url = attach['photo']['sizes'][-1]['url']
                img1 = io.imread(url)
                img = np.array(cv2.resize(img1, (224, 224)))
                img = img / 255.
                img = np.expand_dims(img, axis=0)
                img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
                p = model.predict(img)
                print("Model output: ", p)
                if p > 0.5:
                    vk.messages.send(chat_id=event.chat_id, message="Warning! This is a religious image", random_id= event.obj.random_id)

                    text, label = meme_info(img1)
                    if len(text)>0:
                        vk.messages.send(chat_id=event.chat_id, message="Oh man, this is looks like a meme!",
                                         random_id= event.obj.random_id)
                else:
                    vk.messages.send(chat_id=event.chat_id, message="Definitely not a religious image",
                                     random_id= event.obj.random_id)


if __name__ == '__main__':
    main()