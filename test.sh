# independently trained model
# (rgb+b1) band groups
python test.py --band-groups rgb --model-path output/RGB_LB/best_model.pth --output-dir output/RGB_LB  --test-folder test-90% # LB
python test.py --band-groups rgb --model-path output/RGB_UB/best_model.pth --output-dir output/RGB_UB  --test-folder test-90% # upper bound
python test.py --band-groups rgb --model-path output/RGB_ADG_AUTO/best_model.pth --output-dir output/RGB_ADG_AUTO  --test-folder test-90% --dg adv # ADG_AUTO (auto weighted loss)
python test.py --band-groups rgb --model-path output/RGB_CORAL_AUTO/best_model.pth --output-dir output/RGB_CORAL_AUTO  --test-folder test-90% --dg coral # CORAL_AUTO (auto weighted loss)
python test.py --band-groups rgb --model-path output/RGB_MIXSTYLE/best_model.pth --output-dir output/RGB_MIXSTYLE  --test-folder test-90% --dg mixstyle # MIXSTYLE
python test.py --band-groups rgb --model-path output/RGB_IRM_AUTO/best_model.pth --output-dir output/RGB_IRM_AUTO  --test-folder test-90% # IRM_AUTO (auto weighted loss)

# (rgb+b1+nir) band groups
python test.py --band-groups rgb nir --model-path output/RGB+NIR_LB/best_model.pth --output-dir output/RGB+NIR_LB  --test-folder test-90% # LB
python test.py --band-groups rgb nir --model-path output/RGB+NIR_UB/best_model.pth --output-dir output/RGB+NIR_UB  --test-folder test-90% # upper bound
python test.py --band-groups rgb nir --model-path output/RGB+NIR_ADG_AUTO/best_model.pth --output-dir output/RGB+NIR_ADG_AUTO  --test-folder test-90% --dg adv # ADG_AUTO (auto weighted loss)
python test.py --band-groups rgb nir --model-path output/RGB+NIR_CORAL_AUTO/best_model.pth --output-dir output/RGB+NIR_CORAL_AUTO  --test-folder test-90% --dg coral # CORAL_AUTO (auto weighted loss)
python test.py --band-groups rgb nir --model-path output/RGB+NIR_MIXSTYLE/best_model.pth --output-dir output/RGB+NIR_MIXSTYLE  --test-folder test-90% --dg mixstyle # MIXSTYLE
python test.py --band-groups rgb nir --model-path output/RGB+NIR_IRM_AUTO/best_model.pth --output-dir output/RGB+NIR_IRM_AUTO  --test-folder test-90% # IRM_AUTO (auto weighted loss)

# (rgb+b1)+nir+swir band groups
python test.py --band-groups rgb nir swir --model-path output/RGB+NIR+SWIR_LB/best_model.pth --output-dir output/RGB+NIR+SWIR_LB  --test-folder test-90% # LB
python test.py --band-groups rgb nir swir --model-path output/RGB+NIR+SWIR_UB/best_model.pth --output-dir output/RGB+NIR+SWIR_UB  --test-folder test-90% # upper bound
python test.py --band-groups rgb nir swir --model-path output/RGB+NIR+SWIR_ADG_AUTO/best_model.pth --output-dir output/RGB+NIR+SWIR_ADG_AUTO  --test-folder test-90% --dg adv # ADG_AUTO (auto weighted loss)
python test.py --band-groups rgb nir swir --model-path output/RGB+NIR+SWIR_CORAL_AUTO/best_model.pth --output-dir output/RGB+NIR+SWIR_CORAL_AUTO  --test-folder test-90% --dg coral # CORAL_AUTO (auto weighted loss)
python test.py --band-groups rgb nir swir --model-path output/RGB+NIR+SWIR_MIXSTYLE/best_model.pth --output-dir output/RGB+NIR+SWIR_MIXSTYLE  --test-folder test-90% --dg mixstyle # MIXSTYLE
python test.py --band-groups rgb nir swir --model-path output/RGB+NIR+SWIR_IRM_AUTO/best_model.pth --output-dir output/RGB+NIR+SWIR_IRM_AUTO  --test-folder test-90% # IRM_AUTO (auto weighted loss)

###########################################################################################################################################################################################################################################################
###########################################################################################################################################################################################################################################################
###########################################################################################################################################################################################################################################################
## pratrained models

# (rgb+b1)+nir band groups
python test.py --band-groups rgb nir --model-path output/RGB+NIR_LB_pretrained/best_model.pth --output-dir output/RGB+NIR_LB_pretrained  --test-folder test-90% # LB
python test.py --band-groups rgb nir --model-path output/RGB+NIR_UB_pretrained/best_model.pth --output-dir output/RGB+NIR_UB_pretrained  --test-folder test-90% # upper bound
python test.py --band-groups rgb nir --model-path output/RGB+NIR_ADG_AUTO_pretrained/best_model.pth --output-dir output/RGB+NIR_ADG_AUTO_pretrained  --test-folder test-90% --dg adv # ADG_AUTO (auto weighted loss)
python test.py --band-groups rgb nir --model-path output/RGB+NIR_CORAL_AUTO_pretrained/best_model.pth --output-dir output/RGB+NIR_CORAL_AUTO_pretrained  --test-folder test-90% --dg coral # CORAL_AUTO (auto weighted loss)
python test.py --band-groups rgb nir --model-path output/RGB+NIR_MIXSTYLE_pretrained/best_model.pth --output-dir output/RGB+NIR_MIXSTYLE_pretrained  --test-folder test-90% --dg mixstyle # MIXSTYLE
python test.py --band-groups rgb nir --model-path output/RGB+NIR_IRM_AUTO_pretrained/best_model.pth --output-dir output/RGB+NIR_IRM_AUTO_pretrained  --test-folder test-90% # IRM_AUTO (auto weighted loss)

# (rgb+b1)+nir+swir band groups
python test.py --band-groups rgb nir swir --model-path output/RGB+NIR+SWIR_LB_pretrained/best_model.pth --output-dir output/RGB+NIR+SWIR_LB_pretrained  --test-folder test-90% # LB
python test.py --band-groups rgb nir swir --model-path output/RGB+NIR+SWIR_UB_pretrained/best_model.pth --output-dir output/RGB+NIR+SWIR_UB_pretrained  --test-folder test-90% # upper bound
python test.py --band-groups rgb nir swir --model-path output/RGB+NIR+SWIR_ADG_AUTO_pretrained/best_model.pth --output-dir output/RGB+NIR+SWIR_ADG_AUTO_pretrained  --test-folder test-90% --dg adv # ADG_AUTO (auto weighted loss)
python test.py --band-groups rgb nir swir --model-path output/RGB+NIR+SWIR_CORAL_AUTO_pretrained/best_model.pth --output-dir output/RGB+NIR+SWIR_CORAL_AUTO_pretrained  --test-folder test-90% --dg coral # CORAL_AUTO (auto weighted loss)
python test.py --band-groups rgb nir swir --model-path output/RGB+NIR+SWIR_MIXSTYLE_pretrained/best_model.pth --output-dir output/RGB+NIR+SWIR_MIXSTYLE_pretrained  --test-folder test-90% --dg mixstyle # MIXSTYLE
python test.py --band-groups rgb nir swir --model-path output/RGB+NIR+SWIR_IRM_AUTO_pretrained/best_model.pth --output-dir output/RGB+NIR+SWIR_IRM_AUTO_pretrained  --test-folder test-90% # IRM_AUTO (auto weighted loss)