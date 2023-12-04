> ## ALCS
> 
> Requeirment
> 1.   python==3.7.0
> 2.   numpy==1.26.1
> 3.   matplotlib==3.5.3
> 
> To train policies, 
> 
>     python main.py
>
> You can also change the arguments '--world' and '--task' to select the task you want to learn. For example, the following commands will cover the training of 8 tasks in the paper,
> 
>     python main.py --world=office --task=fg        
>     python main.py --world=office --task=efg
>     python main.py --world=office --task=c4
>     python main.py --world=office --task=bonus
>     python main.py --world=craft --task=plant
>     python main.py --world=craft --task=bridge
>     python main.py --world=craft --task=bed
>     python main.py --world=craft --task=gem




