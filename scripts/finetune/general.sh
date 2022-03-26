
containerID=$1
task=$2
mode=$3
prompt=$4
bot=$5
type=$6
exp_name=$7
end_batch=$8
sample_time=$9
tags=${10}
init_step=${11}
save_path=${12}
model=${13}
extra_label=${14}
save_interval=${15}

echo "task name = ${2},"
echo "mode= ${3},"
echo "prompt = ${4},"
echo "bot = ${5},"
echo "type = ${6},"
echo "exp_name = ${7},"
echo "end_batch = ${8},"
echo "sample_time = ${9},"
echo "wandb tag = ${10},"
echo "init step = ${11},"
echo "save path = ${12},"
echo "model = ${13},"
echo "extra_label = ${extra_label},"
echo "save_interval = ${15}"


CUDA_VISIBLE_DEVICES=0 python main.py \
                        --task $task \
                        --mode $mode \
                        --prompt $prompt \
                        --bot $bot \
                        --type $type \
                        --exp_name $exp_name \
                        --end_batch end_batch \
                        --sample_time $sample_time \
                        --tags $tags \
                        --init_step $init_step \
                        --save_path $save_path \
                        --model $model \
                        --extra_label $extra_label \
                        --save_interval $save_interval