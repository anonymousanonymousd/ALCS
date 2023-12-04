> ## ALCS
> 
> Requeirment
> 1.   python==3.7.0
> 2.   numpy==1.26.1
> 3.   gym==0.15.7
> 4.   matplotlib==3.5.3
> 
> To train policies, 
> 
>     python main.py
>
> To reproduce the result in paper, you may also change the arguments '--world' and '--task' to select the task you want to learn.
>
> For OfficeWorld doamin, the following commends are for 'Coffee', 'Coffee and Mail', 'Collecting' and 'Bonus':
> 
>     python main.py --world=office --task=fg
>     python main.py --world=office --task=efg
>     python main.py --world=office --task=c4
>     python main.py --world=office --task=bonus
>
> Similarly, for the rest 4 tasks on MineCraft domain:
> 
>     python main.py --world=craft --task=plant
>     python main.py --world=craft --task=bridge
>     python main.py --world=craft --task=bed
>     python main.py --world=craft --task=gem
>
> 
> To plot the learning results shown in the paper, you may need command:
> 
>     cd show_results
>     python show_fg_ex2.py
>     python show_ab_ex2.py
> 




