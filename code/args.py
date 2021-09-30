from argparse import ArgumentParser

a_parser = ArgumentParser()
a_parser.add_argument('--embedding_dim', type=int, default=128)
a_parser.add_argument('--hidden_dim', type=int, default=256)
a_parser.add_argument('--use_hallucinated_data', type=bool, default=True)
a_parser.add_argument('--use_curriculum', type=bool, default=True)
a_parser.add_argument('--initial_curriculum', type=int, default=1)

a_parser.add_argument('--char_att', type=str, default='cos-bah')
a_parser.add_argument('--model_orthogonal_init', type=bool, default=False)
a_parser.add_argument('--hiddens_zero_init', type=bool, default=True)

a_parser.add_argument('--opt', type=str, default='adam')
a_parser.add_argument('--lr', type=float, default=0.001) # 001
a_parser.add_argument('--min_lr', type=float, default=0.0000001) # 0.0000001
a_parser.add_argument('--wd', type=float, default=0)#.001) # 00001
a_parser.add_argument('--grad_clip', type=float, default=5.0)

a_parser.add_argument('--sp_step_decay', type=bool, default=True)
a_parser.add_argument('--prob_decay_epoch', type=int, default=0.04) # 0.04
a_parser.add_argument('--prob_decay_steps', type=int, default=400000)
a_parser.add_argument('--train_steps', type=int, default=200000) # 200000
a_parser.add_argument('--full_competence', type=int, default=50000) # 40000
a_parser.add_argument('--n_epochs', type=int, default=10)
a_parser.add_argument('--bs', type=int, default=256) # 256 192
a_parser.add_argument('--c', type=int, default=0)

a_parser.add_argument('--rseed', type=int, default=999)
a_parser.add_argument('--device', type=str, default='cuda:0')

a_parser.add_argument('--do_smoothing', type=bool, default=True)
a_parser.add_argument('--smoothing_value', type=float, default=0.025) # 0.025

# for ablation study
a_parser.add_argument('--scheduled_sampling', type=bool, default=False)
a_parser.add_argument('--label_smoothing', type=bool, default=True)
a_parser.add_argument('--curriculum_learning', type=bool, default=True)
a_parser.add_argument('--multitask_learning', type=bool, default=True)
a_parser.add_argument('--back_inflection', type=bool, default=False)
a_parser.add_argument('--mode', type=str, default='ablation')

a_parser.add_argument('--curriculum_learning_scoring', type=str, default='loss') # norm

args = a_parser.parse_args()
