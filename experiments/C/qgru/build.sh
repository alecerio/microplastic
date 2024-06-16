
# create executable to see the model's results
echo "compile executable to see the model's results ..." 
gcc -o qgru main.c qgru_model.c -O3
echo "compiled correctly"

# create executable to benchmark the model
echo "compile executable to benchmark the model ..."
gcc -o qgru_benchmark framework.c qgru_model.c -O3
echo "compiled correctly"
