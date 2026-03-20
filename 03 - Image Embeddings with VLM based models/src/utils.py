import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import zipfile

from .settings import DATASET_DIR, DATASET_ZIP_FILE, ETC_DIR
from .elastic import knn_search
from elasticsearch import Elasticsearch

import os


def unzip_dataset(dataset_zip_file=DATASET_ZIP_FILE, dataset_dir=DATASET_DIR, etc_dir=ETC_DIR):
    if not os.path.exists(dataset_dir):
        with zipfile.ZipFile(dataset_zip_file, 'r') as data:
            data.extractall(path=etc_dir)
    else:
        print('[INFO] Dataset already unzipped')

def display_knn(knn_items, ref_item=None, src_img_path=None, dataset_dir: str = DATASET_DIR):
    # size of images in the grid
    fig = plt.figure(figsize=(16, 3))

    # get image of the query item and display it in the matplotlib grid
    if ref_item is not None:
        ref_item_img = mpimg.imread(os.path.join(dataset_dir, ref_item['imPath']))
    elif src_img_path is not None:
        ref_item_img = mpimg.imread(src_img_path)
    else:
        raise ValueError('Expected ref_item or src_img_path, got None for both.')

    axis = list()
    axis.append(fig.add_subplot(2, 10, 1))

    # display the query item
    axis[0].imshow(ref_item_img)
    plt.axis("off")

    # display the knn items: loop over the knn items and display their images
    for i, knn_item in enumerate(knn_items):
        knn_img = os.path.join(dataset_dir, knn_item['_source']['imPath'])
        axis.append(fig.add_subplot(2, 10, i + 11))
        axis[i + 1].imshow(mpimg.imread(knn_img))
        plt.axis("off")

    # display the grid
    plt.show()

def make_recommendations(
    es: Elasticsearch,
    item_id: int,
    index_name: str="items",
    model:str ="resnet",
    k=10,
    num_candidates=100,
    apply_filter=False
):
    ref_item, knn_items = knn_search(
        es=es,
        item_id=item_id,
        index_name=index_name,
        model=model,
        k=k,
        num_candidates=num_candidates,
        apply_filter=apply_filter
    )
    display_knn(knn_items, ref_item=ref_item)