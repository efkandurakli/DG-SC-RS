# rgb+b1 band groups
python train.py  --output-dir output/RGB_LB --band-groups rgb  --train-folders train --use-deterministic-algorithms --patience 200 --max-epochs 200 --batch-size 256 --lr-step-size 50 --lr 0.1 # lower bound
python train.py  --output-dir output/RGB_UB --band-groups rgb  --train-folders train test-10% --use-deterministic-algorithms --patience 200 --max-epochs 200 --batch-size 256 --lr-step-size 50 --lr 0.1 # upper bound
python train_dg.py  --output-dir output/RGB_DG --band-groups rgb  --train-folders train --use-deterministic-algorithms --patience 200 --max-epochs 200 --batch-size 256 --lr-step-size 50 --lr 0.1 # domain generalization

#rgb+b1+nir band groups
python train.py  --output-dir output/RGB+NIR_LB --band-groups rgb nir --train-folders train --use-deterministic-algorithms --patience 200 --max-epochs 200 --batch-size 256 --lr-step-size 50 --lr 0.1 # lower bound
python train.py  --output-dir output/RGB+NIR_UB --band-groups rgb nir --train-folders train test-10% --use-deterministic-algorithms --patience 200 --max-epochs 200 --batch-size 256 --lr-step-size 50 --lr 0.1 # upper bound
python train_dg.py  --output-dir output/RGB+NIR_DG --band-groups rgb nir --train-folders train --use-deterministic-algorithms --patience 200 --max-epochs 200 --batch-size 256 --lr-step-size 50 --lr 0.1 # domain generalization

# rgb+b1+nir+swir band groups
python train.py  --output-dir output/RGB+NIR+SWIR_LB --band-groups rgb nir swir --train-folders train --use-deterministic-algorithms --patience 200 --max-epochs 200 --batch-size 256 --lr-step-size 50 --lr 0.1 # lower bound
python train.py  --output-dir output/RGB+NIR+SWIR_UB --band-groups rgb nir swir --train-folders train test-10% --use-deterministic-algorithms --patience 200 --max-epochs 200 --batch-size 256 --lr-step-size 50 --lr 0.1 # upper bound
python train_dg.py  --output-dir output/RGB+NIR+SWIR_DG --band-groups rgb nir swir --train-folders train --use-deterministic-algorithms --patience 200 --max-epochs 200 --batch-size 256 --lr-step-size 50 --lr 0.1 # domain generalization
