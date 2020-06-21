NUM_GENERATIONS=999
NUM_THREADS=50
BEST_OF_THE_BEST=5

bold=$(tput bold)

rm -r -f ./elites/*
rm -r -f ./intermediate_solutions/*
rm -r -f ./running_mean/*
rm -r -f ./params/*

overall_start=$SECONDS
for ((i=1; i <= $NUM_GENERATIONS; i++))
do
  start=$SECONDS
  python prep_running_means.py
  echo ================================================================
  echo
  echo -e "\033[1mGeneration $i\033[0m"
  echo
  for ((j=1; j <= $NUM_THREADS; j++))
  do
    python main.py $i $j & done
  wait
rm -r -f ./elites/*
cp -R ./intermediate_solutions/. ./elites/
rm -r -f ./intermediate_solutions/*
python keep_best_elites.py $BEST_OF_THE_BEST

echo
echo "Completed in $(( SECONDS - start )) seconds"
echo
echo ================================================================
done
wait

python save_best_solution.py
rm -r -f ./params/*
rm -r -f ./elites/*
echo
echo "Algorithm ran in $(( SECONDS - overall_start )) seconds"
echo
