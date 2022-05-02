# cg_gan

รองรับการกำหนด property ได้หลายตัว และ property classes

นำ Hydra + pllighnting + wandb ออกไปหมดแล้ว แต่ยังคงใช้ omegaconf ในการ config file .yaml

ใช้ dataset โดยการ unzip ที่ cdvae/data/mp_20/mp_20.zip จะได้ dataset ที่ processed เป็นกราฟแล้ว

มีการปรับปรุงในส่วนของ gnn ที่ BesselBasisLayer เพื่อให้รันได้ เนื่องจากมีปัญหาเรื่อง inplace operation

เพื่อความง่ายสามารถใช้ python virtual environment ได้ จะได้ไม่มีผลต่อ root

1. ลง python

sudo apt update

sudo apt install python3.8 python3.8-dev python3.8-venv

หรือ

sudo apt install python3.9 python3.9-dev python3.9-venv

2. สร้าง virtual environment

python3.8 -m venv venv_cdvae

หรือ

python3.9 -m venv venv_vcdvae


3. install package ที่เกี่ยวข้องใน virtual environment

source venv_cdvae/bin/activate

python -m ensurepip --upgrade

pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html

pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.1+cu111.html

pip install torch-sparse -f https://data.pyg.org/whl/torch-1.10.1+cu111.html

pip install torch-geometric

pip install pymatgen

pip install omegaconf

pip install smact

pip install p-tqdm

4. รันโค้ด

4.1 สำหรับ train

source vcdvae/bin/activate

cd /content/cg_gan/cdvae

python train.py --config_path ./cdvae/conf/default_class.yaml --output_path ./cdvae/outputs/output1 --predict_property True --predict_property_class True

ถ้าไม่ต้องการใช้งาน predict_property และ predict_property_class ให้นำออกไป


4.2 สำหรับ test

source vcdvae/bin/activate

cd /content/cg_gan/cdvae

python test.py --model_path ./outputs/output1


4.3 สำหรับ evaluate

source vcdvae/bin/activate

cd /content/cg_gan/cdvae


gen จากการสุ่ม z

python evaluate.py --model_path /content/cg_gan/cdvae/outputs/output1 --tasks gen


gen จาก optimization property ที่กำหนด ในชุด mp_20_class มี 3 ตัว คือ formation_energy_per_atom, band_gap  และ e_above_hull
เลือก optimization ได้ทีละตัว โดยผลจะเป็นค่าเฉลี่ยของ parameters ตัวดังกล่าวใน dataset

python evaluate.py --model_path /content/cg_gan/cdvae/outputs/output1 --tasks opt --prop band_gap

gen จากการเลือก class ที่ใส่เข้าไป ในชุด mp_20_class มี 3 ตัว คือ Class_formation_energy_per_atom (มี 0 - 6), Class_band_gap (มี 0 - 4)  และ Class_e_above_hull (มี 0 - 5)
โดยการใช้ --prop_classes และเลือก class เรียงตามลำดับที่กำหนดใน config file เช่น 010 คือ formation_energy class 0, band gap class 1, e_above_hull class 0

python evaluate.py --model_path /content/cg_gan/cdvae/outputs/output4 --tasks gen_classes --prop_classes 010


argument ต่างๆ ที่สามารถเซ็ตได้ดูได้จากในโค้ดโดยตรง เช่น จำนวนที่ gen 

รายละเอียด property ดูได้จาก config file

สำหรับ opt และ gen_classes สามารถเพิ่ม --start_from None ได้เพื่อให้สุ่ม z ขึ้นมาเอง กรณีปกติจะ encode จากข้อมูลจริง 100 ตัวแรก ใน testset เป็น z และใช้การ optimize เพื่อปรับ z ให้ได้ property ตามต้องการ