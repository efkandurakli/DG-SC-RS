# rgb+b1 band groups
python test.py  --band-groups rgb  --model-path output/RGB_LB/best_model.pth --output-dir output/RGB_LB --test-folder test-90%  # lower bound
python test.py  --band-groups rgb  --model-path output/RGB_UB/best_model.pth --output-dir output/RGB_UB --test-folder test-90% # upper bound
python test.py  --band-groups rgb  --model-path output/RGB_DG/best_model.pth --output-dir output/RGB_DG --test-folder test-90% --dg # domain generalization

#rgb+b1+nir band groups
python test.py  --band-groups rgb nir  --model-path output/RGB+NIR_LB/best_model.pth --output-dir output/RGB+NIR_LB --test-folder test-90%  # lower bound
python test.py  --band-groups rgb nir  --model-path output/RGB+NIR_UB/best_model.pth --output-dir output/RGB+NIR_UB --test-folder test-90% # upper bound
python test.py  --band-groups rgb nir  --model-path output/RGB+NIR_DG/best_model.pth --output-dir output/RGB+NIR_DG --test-folder test-90% --dg # domain generalization

# rgb+b1+nir+swir band groups
python test.py  --band-groups rgb nir swir --model-path output/RGB+NIR+SWIR_LB/best_model.pth --output-dir output/RGB+NIR+SWIR_LB --test-folder test-90%  # lower bound
python test.py  --band-groups rgb nir swir  --model-path output/RGB+NIR+SWIR_UB/best_model.pth --output-dir output/RGB+NIR+SWIR_UB --test-folder test-90% # upper bound
python test.py  --band-groups rgb nir swir  --model-path output/RGB+NIR+SWIR_DG/best_model.pth --output-dir output/RGB+NIR+SWIR_DG --test-folder test-90% --dg # domain generalization

