#!/bin/bash
##
## ntasks = quantidade de nucleos
#SBATCH --ntasks=3
# 
## time = quantidade de tempo solicitado
#SBATCH --time=360:00:00
#
## Configura o envio de e-mail quando o job for cancelado/finalizado.
## Substitua "root" por seu endereco de e-mail.
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=clecio@debom.com.br
#
## Nome do job. Aparece na saida do comando 'squeue'.
## E recomendado, mas nao necesssario, que o nome do job
## seja o mesmo que o nome do arquivo de input
#SBATCH --job-name=bmoend
#SBATCH --partition=cosmoobs
echo -e "\n## Job iniciado em $(date +'%d-%m-%Y as %T') #####################\n"

## O nome dos arquivos de input e output sao baseados no nome do job.
## Observe que nao e obrigatorio esta forma de nomear os arquivos.
INP="lenstool_in.par"
##$SLURM_JOB_NAME".inp"

## Informacoes do job impressos no arquivo de saida.
echo -e "\n## Jobs ativos de $USER: \n"
squeue -a -u $USER
echo -e "\n## Node de execucao do job: $(hostname -s) \n"
echo -e "\n## Numero de tarefas para este job: $SLURM_NTASKS \n"

#########################################
##-------  Inicio do trabalho ----- #
#########################################

## Configura o ambiente de execucao do software.
##module load softwares/lenstool/r1398-gnu-5.3
cd $SLURM_SUBMIT_DIR
## Numero de threads de acordo com o que foi alocado em "--ntasks"
export OMP_NUM_THREADS=$SLURM_NTASKS

## Informacoes sobre o ambiente de execucao impressos no arquivo de saida.
echo -e "\n## Diretorio de submissao do job:   $SLURM_SUBMIT_DIR \n"
echo -e "\n## Diretorio de scratch do job:     $SCRATCH \n"
echo -e "\n## Arquivo de input:                $INP \n"

## Execucao do software
source activate gwh0
## nohup python kn_strategy_sims.py > out20211223_Mblue_nmo_0_50.txt
python kn_strategy_sims.py
