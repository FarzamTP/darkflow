from matplotlib import pyplot as plt
from generate_xml import write_xml
import numpy as np
import urllib.request as req
import requests
import ast
import os
import cv2


def draw_random_image():
    random_index = np.random.randint(0, len(os.listdir('./dataset')) - 1)
    img = cv2.imread('./dataset/image{}.jpg'.format(str(random_index)))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for i in range(len(tl[random_index])):
        img = cv2.rectangle(img, tl[random_index][i], br[random_index][i], (0, 255, 0), 5)

    plt.imshow(img)
    plt.title(random_index)
    plt.show(random_index)
    return


def download_image(urls_list):
    for idx, url in enumerate(urls_list):
        if not 'image{}.jpg'.format(str(idx)) in os.listdir('./dataset'):
            if requests.get(url).ok:
                req.urlretrieve(url, './dataset/image{}.jpg'.format(str(idx)))
                print("[INFO]: %s of %s downloaded." % (str(idx), str(len(urls_list))))
            else:
                print("[ERROR]: Url {} is not reachable.".format(url))
        else:
            print('[INFO]: Image{}.jpg already downloaded.'.format(str(idx)))
    return


def read_json_file(path):
    with open(path) as file:
        retrieved_data = file.readlines()
    return retrieved_data


def write_image_urls_to_file(urls_list):
    with open('./image_urls.txt', 'w') as f:
        for url in urls_list:
            f.write(url)
            f.write('\n')
    return


def generate_tl_br_list(data_dict, data_urls):
    img_tl_list = []
    img_br_list = []

    data_dict = ast.literal_eval(data_dict.replace('null', '""'))
    data_urls.append(data_dict.get('content'))

    for i in range(len(data_dict.get('annotation'))):
        tl_x = int(data_dict.get('annotation')[i].get('points')[0].get('x') * data_dict.get('annotation')[0].get(
            'imageWidth'))
        tl_y = int(data_dict.get('annotation')[i].get('points')[0].get('y') * data_dict.get('annotation')[0].get(
            'imageHeight'))

        br_x = int(data_dict.get('annotation')[i].get('points')[1].get('x') * data_dict.get('annotation')[0].get(
            'imageWidth'))
        br_y = int(data_dict.get('annotation')[i].get('points')[1].get('y') * data_dict.get('annotation')[0].get(
            'imageHeight'))

        tl = (tl_x, tl_y)
        br = (br_x, br_y)

        img_tl_list.append(tl)
        img_br_list.append(br)
    return data_dict, img_tl_list, img_br_list


def fetch_image_urls(path):
    data_dictionary = []
    data_urls = []
    tl_list = []
    br_list = []
    raw_data = read_json_file(path)
    for idx, data_dict in enumerate(raw_data):

        data_dict, img_tl_list, img_br_list = generate_tl_br_list(data_dict=data_dict, data_urls=data_urls)

        data_dictionary.append(data_dict)

        tl_list.append(img_tl_list)
        br_list.append(img_br_list)

        objects = []
        for i in range(len(data_dict.get('annotation'))):
            objects.append(data_dict.get('annotation')[i].get('label')[0])

        write_xml(folder='./dataset', img_path='./dataset/image{}.jpg'.format(str(idx)),
                  img_name='image{}.jpg'.format(str(idx)), objects=objects,
                  tl=tl_list[idx], br=br_list[idx], savedir='./annotation')
        
    print("[INFO]: Xml files has been generated!")
        
    if os.path.getsize('./image_urls.txt') == 0:
        write_image_urls_to_file(data_urls)
    return data_dictionary, data_urls, tl_list, br_list


if __name__ == '__main__':
    JSON_FILE_PATH = "./face_detection.json"
    data_dictionary_list, images_urls_list, tl, br = fetch_image_urls(JSON_FILE_PATH)
    print("[INFO]: Data read from Json file and converted to dictionaries.")
    if len(os.listdir('./dataset')) != len(images_urls_list):
        download_image(images_urls_list)
        print("[INFO]: Images downloaded and ready to process.")
    else:
        print("[INFO]: Images ready for processing.")

    # draw_random_image()
