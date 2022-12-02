import subprocess
import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, required=True)
parser.add_argument('--a', type=str, required=True)
parser.add_argument('--loop', type=bool, required=True)
def main():
    args = parser.parse_args()
    max_camera_angles = args.a
    loop_scenes = args.loop
    scene_iter = 0 
    print(args.path)
    for subdir, dirs, files in os.walk(args.path):
        while (True):
            for file in files:
                if (not file.endswith(".txt")):
                    continue
                subprocess.call([".\\bin\\Release\\cis565_final.exe", os.path.join(subdir, file), max_camera_angles, str(scene_iter)], shell=True)
        if (not loop_scenes):
            break
        else:
            scene_iter += 1




if __name__ == "__main__":
   main()