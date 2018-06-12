.PHONY: clean data lint requirements sync_data_to_s3 sync_data_from_s3

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
BUCKET = [OPTIONAL] your-bucket-for-syncing-data (do not include 's3://')
PROFILE = default
PROJECT_NAME = titanic
PYTHON_INTERPRETER = python3

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python Dependencies
requirements: test_environment requirements.txt
	pip install -U pip setuptools wheel
	pip install -r requirements.txt

## Download Training Data
data/raw/train.csv : titanic_src/data/make_dataset.py
	$(PYTHON_INTERPRETER) titanic_src/data/make_dataset.py download_data data/raw/train.csv

## Download Testing Data
data/raw/test.csv : titanic_src/data/make_dataset.py
	$(PYTHON_INTERPRETER) titanic_src/data/make_dataset.py download_data data/raw/test.csv

## Download Submission Example
data/raw/gender_submission.csv : titanic_src/data/make_dataset.py
	$(PYTHON_INTERPRETER) titanic_src/data/make_dataset.py download_data data/raw/gender_submission.csv

## Make feature creation pipeline
models/new_features_pipeline.pkl : data/raw/train.csv titanic_src/features/build_features.py
	$(PYTHON_INTERPRETER) titanic_src/features/build_features.py fit_new_features --input_path data/raw/train.csv models/new_features_pipeline.pkl

## Create new features in training data
data/interim/added_features_train.csv : data/raw/train.csv models/new_features_pipeline.pkl
	$(PYTHON_INTERPRETER) titanic_src/features/build_features.py transform_new_features data/raw/train.csv data/interim/added_features_train.csv --pipeline_input_path models/new_features_pipeline.pkl

## Create new features in testing data
data/interim/added_features_test.csv : data/raw/test.csv models/new_features_pipeline.pkl
	$(PYTHON_INTERPRETER) titanic_src/features/build_features.py transform_new_features data/raw/test.csv data/interim/added_features_test.csv --pipeline_input_path models/new_features_pipeline.pkl

## Add new features
add_features : data/interim/added_features_test.csv data/interim/added_features_train.csv

## Make impute pipeline
models/impute_pipeline.pkl: data/interim/added_features_train.csv titanic_src/features/transform_features.py
	$(PYTHON_INTERPRETER) titanic_src/features/transform_features.py fit_imputer --input_path data/interim/added_features_train.csv models/impute_pipeline.pkl

## Impute missing training data
data/interim/imputed_train.csv : data/interim/added_features_train.csv models/impute_pipeline.pkl
	$(PYTHON_INTERPRETER) titanic_src/features/transform_features.py transform_imputer  data/interim/added_features_train.csv data/interim/imputed_train.csv --pipeline_input_path models/impute_pipeline.pkl --add_target

## Impute missing testing data
data/interim/imputed_test.csv : data/interim/added_features_test.csv models/impute_pipeline.pkl
	$(PYTHON_INTERPRETER) titanic_src/features/transform_features.py transform_imputer  data/interim/added_features_test.csv data/interim/imputed_test.csv --pipeline_input_path models/impute_pipeline.pkl

## Make binning pipeline
models/bins_pipeline.pkl: data/interim/imputed_train.csv titanic_src/features/build_features.py
	$(PYTHON_INTERPRETER) titanic_src/features/build_features.py fit_bins --input_path data/interim/imputed_train.csv models/bins_pipeline.pkl

## Create new features in training data
data/interim/bins_train.csv : data/interim/imputed_train.csv models/bins_pipeline.pkl
	$(PYTHON_INTERPRETER) titanic_src/features/build_features.py transform_bins data/interim/imputed_train.csv data/interim/bins_train.csv --pipeline_input_path models/bins_pipeline.pkl

## Create new features in testing data
data/interim/bins_test.csv : data/interim/imputed_test.csv models/bins_pipeline.pkl
	$(PYTHON_INTERPRETER) titanic_src/features/build_features.py transform_bins data/interim/imputed_test.csv data/interim/bins_test.csv --pipeline_input_path models/bins_pipeline.pkl

## Onehot encode training data
data/processed/final_train.csv : data/interim/bins_train.csv titanic_src/features/transform_features.py
	$(PYTHON_INTERPRETER) titanic_src/features/transform_features.py fit_transform_encoder  data/interim/bins_train.csv data/processed/final_train.csv --add_target

## Onehot encode testing data
data/processed/final_test.csv : data/interim/bins_test.csv titanic_src/features/transform_features.py
	$(PYTHON_INTERPRETER) titanic_src/features/transform_features.py fit_transform_encoder  data/interim/bins_test.csv data/processed/final_test.csv


## Process all data
all_data : data/processed/final_train.csv data/processed/final_test.csv

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8
lint:
	flake8 src

## Upload Data to S3
sync_data_to_s3:
ifeq (default,$(PROFILE))
	aws s3 sync data/ s3://$(BUCKET)/data/
else
	aws s3 sync data/ s3://$(BUCKET)/data/ --profile $(PROFILE)
endif

## Download Data from S3
sync_data_from_s3:
ifeq (default,$(PROFILE))
	aws s3 sync s3://$(BUCKET)/data/ data/
else
	aws s3 sync s3://$(BUCKET)/data/ data/ --profile $(PROFILE)
endif

## Set up python interpreter environment
create_environment:
ifeq (True,$(HAS_CONDA))
		@echo ">>> Detected conda, creating conda environment."
ifeq (3,$(findstring 3,$(PYTHON_INTERPRETER)))
	conda create --name $(PROJECT_NAME) python=3
else
	conda create --name $(PROJECT_NAME) python=2.7
endif
		@echo ">>> New conda env created. Activate with:\nsource activate $(PROJECT_NAME)"
else
	@pip install -q virtualenv virtualenvwrapper
	@echo ">>> Installing virtualenvwrapper if not already intalled.\nMake sure the following lines are in shell startup file\n\
	export WORKON_HOME=$$HOME/.virtualenvs\nexport PROJECT_HOME=$$HOME/Devel\nsource /usr/local/bin/virtualenvwrapper.sh\n"
	@bash -c "source `which virtualenvwrapper.sh`;mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER)"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
endif

## Test python environment is setup correctly
test_environment: test_environment.py
	$(PYTHON_INTERPRETER) test_environment.py > environment_passed.txt

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := show-help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: show-help
show-help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
