# function to verify if DatasetCatalog is loaded correctly
def test_dataset(path):
    """
    Test if detectron2_dataset function is correct.

    Params:
        path = path to directory of images and labels 
        (aka train or test directory)
    """
    # get train/test type from path
    img_type = "train" if path.find("train") != -1 else "test"

    # get dataset and load metadata
    data = detectron2_dataset(path)
    DatasetCatalog.register("snake_" + img_type, lambda d=img_type: detectron2_dataset(platform + img_type + "/"))
    MetadataCatalog.get("snake_" + img_type).set(thing_classes=["snake"])
    snake_metadata = MetadataCatalog.get("snake_" + img_type)

    # get 3 random images and check if they're correct
    for d in data:
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=snake_metadata, scale=0.7)
        vis = visualizer.draw_dataset_dict(d)
        cv2_show_img("Sneks", vis.get_image()[:, :, ::-1])
