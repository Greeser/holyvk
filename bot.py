import vk_api
from vk_api.bot_longpoll import VkBotLongPoll, VkBotEventType
from skimage import io
import numpy as np
import cv2
from keras.models import load_model

model = load_model("weights.hdf5")
model.summary()
#model = None
def main():
    """ Пример использования bots longpoll
        https://vk.com/dev/bots_longpoll
    """

    vk_session = vk_api.VkApi(token='637ee18f09522ab4af2426e264709932f2e0ffc30ff0a8f601d9014fbfdda76c1bd4ac0e19e611a4bb289')
    vk = vk_session.get_api()
    longpoll = VkBotLongPoll(vk_session, '174379976')

    for event in longpoll.listen():

        if event.type == VkBotEventType.MESSAGE_NEW:
            print('Новое сообщение:')

            print('Для меня от: ', end='')
            print(event.obj)

            if event.obj.attachments:
                attach = event.obj.attachments[0]
                #print(event.obj.attachments)
                url =attach['photo']['sizes'][-1]['url']
                img1 = io.imread(url)
           #     cv2.imshow('lalala', img1)
            #    cv2.waitKey(0)
                img = np.array(cv2.resize(img1, (224, 224)))
                img = img / 255.
                img = np.expand_dims(img, axis=0)

                p = model.predict(img)
                print("pred",p)
                if ( p > 0.5):
                    print('А вот тут оскорбили')
                    vk.messages.send(chat_id=event.chat_id, message="А вот тут икона", random_id= event.obj.random_id)
                else:
                    print("Тут не оскорбили")
                    vk.messages.send(chat_id=event.chat_id, message="Тут не икона", random_id= event.obj.random_id)


            print(event.chat_id)
            vk.messages.send(chat_id=event.chat_id, message="Hello man", random_id= event.obj.random_id)
            print()

        elif event.type == VkBotEventType.MESSAGE_REPLY:
            print('Новое сообщение:')

            print('От меня для: ', end='')

            print(event.obj.peer_id)
            print('Текст:', event.obj.text)
            print()

        elif event.type == VkBotEventType.MESSAGE_TYPING_STATE:
            print('Печатает ', end='')

            print(event.obj.from_id, end=' ')

            print('для ', end='')

            print(event.obj.to_id)
            print()

        elif event.type == VkBotEventType.GROUP_JOIN:
            print(event.obj.user_id, end=' ')

            print('Вступил в группу!')
            print()

        elif event.type == VkBotEventType.GROUP_LEAVE:
            print(event.obj.user_id, end=' ')

            print('Покинул группу!')
            print()

        else:
            print(event.type)
            print()


if __name__ == '__main__':
    main()