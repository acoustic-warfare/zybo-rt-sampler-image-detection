from run_object_oriented import yolo_model

def main(): 
    # Path to the trained YOLO model and the source video
    model_path = "model/best.pt"
    # source = "../test_footage/video/drone_video.mp4"
    source = "https://www.youtube.com/watch?v=SJbdUxk8GG0"


    # Create an instance of the yolo_model class
    yolo = yolo_model(model_path)
    


    yolo_model.run_conf_n_inference(yolo, source=source, stream=True, show=True)

    
if __name__ == "__main__":
    main()