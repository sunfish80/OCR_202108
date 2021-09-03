# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
project = 'PaddleOCR'  # 工作项目根目录
sys.path.append(os.getcwd().split(project)[0] + project)
import time
import multiprocessing
import numpy as np

def set_paddle_flags(**kwargs):
    for key, value in kwargs.items():
        if os.environ.get(key, None) is None:
            os.environ[key] = str(value)


# NOTE(paddle-dev): All of these flags should be
# set before `import paddle`. Otherwise, it would
# not take any effect.
set_paddle_flags(
    FLAGS_eager_delete_tensor_gb=0,  # enable GC to save memory
)

from paddle import fluid

# from ppocr.utils.utility import load_config, merge_config
import program
from paddle import fluid
from ppocr.utils.utility import initial_logger
logger = initial_logger()
from ppocr.data.reader_main import reader_main
from ppocr.utils.save_load import init_model
from ppocr.utils.character import CharacterOps
from ppocr.utils.utility import create_module
from ppocr.utils.utility import get_image_file_list
logger = initial_logger()


def main():
    config = program.load_config(FLAGS.config)
    program.merge_config(FLAGS.opt)
    logger.info(config)
    char_ops = CharacterOps(config['Global'])
    config['Global']['char_ops'] = char_ops

    # check if set use_gpu=True in paddlepaddle cpu version
    use_gpu = config['Global']['use_gpu']
    #     check_gpu(use_gpu)

    place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)

    rec_model = create_module(config['Architecture']['function'])(params=config)

    startup_prog = fluid.Program()
    eval_prog = fluid.Program()
    with fluid.program_guard(eval_prog, startup_prog):
        with fluid.unique_name.guard():
            _, outputs = rec_model(mode="test")
            fetch_name_list = list(outputs.keys())
            fetch_varname_list = [outputs[v].name for v in fetch_name_list]
    eval_prog = eval_prog.clone(for_test=True)
    exe.run(startup_prog)

    init_model(config, eval_prog, exe)

    blobs = reader_main(config, 'test')()
    infer_img = config['TestReader']['infer_img']
    infer_list = get_image_file_list(infer_img)
    max_img_num = len(infer_list)
    if len(infer_list) == 0:
        logger.info("Can not find img in infer_img dir.")
    from tqdm import tqdm
    f = open('result.txt',mode='w',encoding='utf8')
    f.write('new_name'+'\t'+'value'+'\n')
    for i in tqdm(range(max_img_num)):
        # print("infer_img:",infer_list[i])
        img,img_path = next(blobs)
        predict = exe.run(program=eval_prog,
                          feed={"image": img},
                          fetch_list=fetch_varname_list,
                          return_numpy=False)

        preds = np.array(predict[0])
        if preds.shape[1] == 1:
            preds = preds.reshape(-1)
            preds_lod = predict[0].lod()[0]
            preds_text = char_ops.decode(preds)
        else:
            end_pos = np.where(preds[0, :] == 1)[0]
            if len(end_pos) <= 1:
                preds_text = preds[0, 1:]
            else:
                preds_text = preds[0, 1:end_pos[1]]
            preds_text = preds_text.reshape(-1)
            preds_text = char_ops.decode(preds_text)

        f.write('{}\t{}\n'.format(os.path.basename(img_path),preds_text))
        # print("\t index:",preds)
        # print("\t word :",preds_text)
    f.close()
    # save for inference model
    # target_var = []
    # for key, values in outputs.items():
    #     target_var.append(values)

    # fluid.io.save_inference_model(
    #     "./output/",
    #     feeded_var_names=['image'],
    #     target_vars=target_var,
    #     executor=exe,
    #     main_program=eval_prog,
    #     model_filename="model",
    #     params_filename="params")


if __name__ == '__main__':
    parser = program.ArgsParser()
    FLAGS = parser.parse_args()
    FLAGS.config = 'configs/rec/rec_r34_vd_none_bilstm_ctc.yml'
    main()
