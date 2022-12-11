import subprocess
import os
def main(argv):
    max_camera_angles = argv[1]
    loop_scenes = argv[2].lower == "true"
    scene_iter = 0 
    for subdir, dirs, files in os.walk(argv[0]):
        for file in files:
            while (True):
                subprocess.call([".\\bin\\Release\\cis565_final.exe", max_camera_angles, scene_iter])
                if (not loop_scenes):
                    break
                scene_iter += 1



if __name__ == "__main__":
   main(sys.argv[1:])