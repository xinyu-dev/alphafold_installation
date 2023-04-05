# AlphaFold Basic Usage


# Run prediction 

## Monomer

1. start a new tmux session with

   ```
   tmux new -s session_name
   ```

   This starts a named tmux session that runs even after you turn off laptop (as long as instance is still running on AWS).  

3. Cd into efs volume (or whichever folder you want to save your results)

   ```
   cd /efs
   ```

4. Optionally, I make a folder specific to my project, for example:

   ```
   mkdir myproject
   ```

6. Copy paste sequence into the a fasta file

   ```
   cd myproject
   vim myfasta.fasta
   ```

   For example:

   ```
   >myseq
   PUTSEQUENCEHERE
   ```

7. Make sure you in `tmux` mode. Then activate environment

   ```
   source activate alphafold
   ```

8. Run

   ```
   python3 /data/alphafold/docker/run_docker.py --fasta_paths=/efs/myproject/myfasta.fasta --output_dir=/efs/myproject --model_preset=monomer
   ```


## Heterodimer

Similar to homodimers, except you put the 2 sequences in 1 fasta file. 

1. For example:
   ```
   >chain1
   CHAINSEQUENCE
   
   >chain2
   CHAINSEQUENCE
   ```

2. Activate env by `source activate alphafold`. Then Run the multimer mode:

   ```
   python3 /data/alphafold/docker/run_docker.py --fasta_paths=/efs/myproject/myfasta.fasta --output_dir=/efs/myproject --model_preset=multimer
   ```

## Homomer (General)

Method is similar, but in the fasta file provide N copies of the same sequence. For example, if the homomer has 3 copies of the same sequence, your input fasta file should be:

```
>sequence_1
<SEQUENCE>
>sequence_2
<SEQUENCE>
>sequence_3
<SEQUENCE>
```

Run command is same as heterodimer. 

## Heteromer (General)

Say we have an A2B3 heteromer, i.e. with 2 copies of `<SEQUENCE A>` and 3 copies of `<SEQUENCE B>`. The input fasta should be:

```
>sequence_1
<SEQUENCE A>
>sequence_2
<SEQUENCE A>
>sequence_3
<SEQUENCE B>
>sequence_4
<SEQUENCE B>
>sequence_5
<SEQUENCE B>
```

Run command is same as heterodimer. 


## Multiple fasta files

Similar as described before, just pass this argument: 

For example:
```
--fasta_paths=multimer1.fasta,multimer2.fasta
```

   