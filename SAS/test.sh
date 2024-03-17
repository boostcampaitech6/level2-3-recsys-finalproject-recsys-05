if  [ "$1" = "dataset" ]; then 
    python -m pytest -v test/test_dataset.py 
elif [ "$1" = "model" ]; then 
    python -m pytest -v test/test_model.py 
elif [ "$1" = "train" ]; then 
    python -m pytest -v test/test_train.py 
elif [ "$1" = "valid" ]; then 
    python -m pytest -v test/test_valid.py 
elif [ "$1" = "inference" ]; then 
    python -m pytest -v test/test_inference.py 
else
    echo "Invalid argument"
fi