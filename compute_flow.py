import cv2
import numpy as np
from tqdm import tqdm
import os
import h5py
import argparse

from rembg import remove
from PIL import Image

def remove_background_video(input_video, output_video):
    input_video_cap = cv2.VideoCapture(input_video)
    width = int(input_video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(input_video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(input_video_cap.get(cv2.CAP_PROP_FPS))
    codec = int(input_video_cap.get(cv2.CAP_PROP_FOURCC))
    total_frames = int(input_video_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_video_writer = cv2.VideoWriter(output_video, codec, fps, (width, height), isColor=True)

    progress_bar = tqdm(range(total_frames), desc="Processing frames", ncols=100)

    for _ in progress_bar:
        ret, frame = input_video_cap.read()
        if not ret:
            break

        pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        output_frame = remove(pil_frame, session=session)
        output_frame = cv2.cvtColor(np.array(output_frame), cv2.COLOR_RGB2BGR)

        output_video_writer.write(output_frame)

    input_video_cap.release()
    output_video_writer.release()


def main(args):
    from rembg import new_session

    session = new_session(gpu=True)


    W, H = args.width, args.height

    if args.remove_background:
      pil_cur_frame = Image.fromarray(cv2.cvtColor(cur_frame, cv2.COLOR_BGR2RGB))
      removed_background_frame = remove(pil_cur_frame, session=session)
      cur_frame = cv2.cvtColor(np.array(removed_background_frame), cv2.COLOR_RGB2BGR)

    input_video = cv2.VideoCapture(args.input_video)
    fps = int(input_video.get(cv2.CAP_PROP_FPS))
    total_frames = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))

    prev_frame = None

    with h5py.File(args.output_file, 'w') as f: pass

    with h5py.File(args.output_file, 'a') as f:
        flow_maps = f.create_dataset('flow_maps', shape=(0, 2, H, W, 2), maxshape=(None, 2, H, W, 2), dtype=np.float16) 

        for ind in tqdm(range(total_frames)):
            if not input_video.isOpened(): break
            ret, cur_frame = input_video.read()
            if not ret: break

            cur_frame = cv2.resize(cur_frame, (W, H))

            if prev_frame is not None:
                next_flow, prev_flow, occlusion_mask = RAFT_estimate_flow(prev_frame, cur_frame)

                flow_maps.resize(ind, axis=0)
                flow_maps[ind-1, 0] = next_flow
                flow_maps[ind-1, 1] = prev_flow

                occlusion_mask = np.clip(occlusion_mask * 0.2 * 255, 0, 255).astype(np.uint8)

                if args.visualize:
                    img_show = cv2.hconcat([cur_frame, occlusion_mask])
                    cv2.imshow('Out img', img_show)
                    if cv2.waitKey(1) & 0xFF == ord('q'): exit()

            prev_frame = cur_frame.copy()

    input_video.release()

    if args.visualize: cv2.destroyAllWindows()

    if args.remove_background:
        os.remove(temp_video_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_video', help="Path to input video file", required=True)
    parser.add_argument('-o', '--output_file', help="Path to output flow file. Stored in *.h5 format", required=True)
    parser.add_argument('-W', '--width', help='Width of the generated flow maps', default=1024, type=int)
    parser.add_argument('-H', '--height', help='Height of the generated flow maps', default=576, type=int)
    parser.add_argument('-v', '--visualize', action='store_true', help='Show proceed images and occlusion maps')
    parser.add_argument('-rb', '--remove_background', action='store_true', help='Remove background using U-2-Net before computing optical flow')
    args = parser.parse_args()

    main(args)
