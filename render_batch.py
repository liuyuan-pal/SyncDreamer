import subprocess

from ldm.base_utils import save_pickle

uids=['6f99fb8c2f1a4252b986ed5a765e1db9','8bba4678f9a349d6a29314ccf337975c','063b1b7d877a402ead76cedb06341681',
      '199b7a080622422fac8140b61cc7544a','83784b6f7a064212ab50aaaaeb1d7fa7','5501434a052c49d6a8a8d9a1120fee10',
      'cca62f95635f4b20aea4f35014632a55','d2e8612a21044111a7176da2bd78de05','f9e172dd733644a2b47a824e202c89d5']

for uid in uids:
    cmds = ['blender','--background','--python','blender_script.py','--',
            '--object_path',f'objaverse_examples/{uid}/{uid}.glb',
            '--output_dir','./training_examples/input','--camera_type','random']
    subprocess.run(cmds)

    cmds = ['blender','--background','--python','blender_script.py','--',
            '--object_path',f'objaverse_examples/{uid}/{uid}.glb',
            '--output_dir','./training_examples/target','--camera_type','fixed']
    subprocess.run(cmds)

save_pickle(uids, f'training_examples/uid_set.pkl')