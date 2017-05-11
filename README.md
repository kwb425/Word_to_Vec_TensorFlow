[![tag][a]][1]
[![release][b]][2]
[![download][c]][3]
# Table of Contents <a name="anchor_main"></a>
---
1. [Running Environment](#anchor_1) <br></br>
2. [data](#anchor_2) <br></br>
3. [log](#anchor_3) <br></br>
4. [scripts](#anchor_4) <br></br>
5. [References](#anchor_ref) <br></br>

## Running Environment <a name="anchor_1"></a> [up](#anchor_main)
Virtualenv Python 2.7 with TensorFlow on it with following packages:

* pip install --upgrade jupyter
* pip install --upgrade pillow
* pip install --upgrade scikit-learn
* pip install --upgrade matplotlib
* pip install --upgrade scipy
* pip install --upgrade xlrd
* pip install --upgrade python\_speech\_features
* pip install --upgrade nltk
* pip install git+http://github.com/kylebgorman/textgrid.git
* pip install --upgrade JPype1
* pip install --upgrade konlpy

## data <a name="anchor_2"></a> [up](#anchor_main)
| Data                | Type            | Input to                 |
| :-------:           | :---:           | :----:                   |   
| crude.txt           | to be organized | input\_organizer.py      | 
| input               | organized       | word2vec\_tensorboard.py | 
| backup\_input\_eng  | backup file     | NULL                     | 
| backup\_input\_nltk | backup file     | NULL                     | 
* Just change the backup file's name to "input (second row from above table)", when feeding them as new input (make sure you also clear all the pre-existing logs before the new run).

## log <a name="anchor_3"></a> [up](#anchor_main)
* checkpoint file sould be altered in a way: 
	
	```
	First, try:
	model_checkpoint_path: "model.ckpt-100000"
	all_model_checkpoint_paths: "model.ckpt-60000"
	all_model_checkpoint_paths: "model.ckpt-70000"
	all_model_checkpoint_paths: "model.ckpt-80000"
	all_model_checkpoint_paths: "model.ckpt-90000"
	all_model_checkpoint_paths: "model.ckpt-100000"
	
	if above does not work, then:
	model_checkpoint_path: "/Your/path/to/log/model.ckpt-100000"
	all_model_checkpoint_paths: "/Your/path/to/log/model.ckpt-60000"
	all_model_checkpoint_paths: "/Your/path/to/log/model.ckpt-70000"
	all_model_checkpoint_paths: "/Your/path/to/log/model.ckpt-80000"
	all_model_checkpoint_paths: "/Your/path/to/log/model.ckpt-90000"
	all_model_checkpoint_paths: "/Your/path/to/log/model.ckpt-100000"	```
	
* events.out.tfevents.1494508700.%s-MacBook.local file should be altered in a way:

	```
	events.out.tfevents.1494508700.YOUR_COMPUTER_NAME-MacBook.local
	```

* labels.tsv is meta data for the embedding (you can make your own if you want to).		

## scripts <a name="anchor_4"></a> [up](#anchor_main)
* This directory "MUST BE" your working directory.
* word2vec\_tensorboard.py: this is main().
* input\_organizer.py: crude.txt -> input
* See the codes for details.

## References <a name="anchor_ref"></a> [top](#anchor_main)
My [blog][4] <br></br>
Email: <kwb425@icloud.com>

<!--Links to addresses, reference Markdowns-->
[1]: https://github.com/kwb425/Word_to_Vec_TensorFlow/tags
[2]: https://github.com/kwb425/Word_to_Vec_TensorFlow/releases
[3]: https://github.com/kwb425/Word_to_Vec_TensorFlow/releases
[4]: http://kwb425.github.io/
<!--Links to images, reference Markdowns-->
[a]: https://img.shields.io/badge/Tag-v1.1-red.svg?style=plastic
[b]: https://img.shields.io/badge/Release-v1.1-green.svg?style=plastic
[c]: https://img.shields.io/badge/Download-Click-blue.svg?style=plastic
