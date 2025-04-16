#!/bin/bash
#SBATCH -A kazmierczak-group
#SBATCH --job-name=PI # Tu nazywasz jakoś swój proces, byle co szczerze mało warte bo i tak po nicku ja znajduje mój task
#SBATCH --time=23:00:00 # dla short to masz max 2h dla long i experimental masz chyba 3-4 dni to jest czas po którym slurm ubja twój proces (zasada jest że nie dajesz maksa bo wtedy do dupy się kolejkują taski a też dajesz takie +2h takiemu maksowi który sprawdziłeś)
#SBATCH --ntasks=1 # tutaj wystarczy 1 zawsze mieć chyba że chcesz multi gpu itp ale zapewne 1 GPU wam wystarczy
#SBATCH --gpus=1 # Jak nie potrzebujesz GPU to wyrzucasz tą linijke
#SBATCH --cpus-per-gpu=8 # Ile cpu na jedno gpu ma być w tym konfigu to po prostu ile cpu chcesz mieć mówiłem żeby dawać zawsze minimum 6-8 bo inaczej kolejkowanie się psuje
#SBATCH --mem=64gb # Ile ram chcesz mieć mamy dużo więc nie musisz dawać mało ale bez przesady
#SBATCH --partition=short # Tutaj podajesz short,long,experimental jedną z tych partycji z której chcesz korzystać shot i long ma A100 short max 1d long dłużej a experimental gorsze GPU  
#SBATCH --mail-type=ALL
#SBATCH --mail-user=01180696@pw.edu.pl

# Debugging flags (optional) --gpus 1
export PYTHONFAULTHANDLER=1

cd /mnt/evafs/groups/kazmierczak-group/Projekt-Interdyscyplinarny # zawsze wpierw kieruje do ścieżki z kodem

source /mnt/evafs/groups/kazmierczak-group/Projekt-Interdyscyplinarny/.venv/bin/activate
python code/train.py slurm_id=${SLURM_JOB_ID} "$@"
