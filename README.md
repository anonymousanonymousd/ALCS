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
> To reproduce the result in paper, you may also change the arguments '--world' and '--task' to select the task you want to learn.
>
> Coffee task for OfficeWorld domain:
> 
>     python main.py --world=office --task=fg
>
> Coffee and Mail task for OfficeWorld domain:
> 
>     python main.py --world=office --task=efg
>
> Collecting task for OfficeWorld domain:
> 
>     python main.py --world=office --task=c4
>
> Bonus task for OfficeWorld domain:
> 
>     python main.py --world=office --task=bonus
>     python main.py --world=craft --task=plant
>     python main.py --world=craft --task=bridge
>     python main.py --world=craft --task=bed
>     python main.py --world=craft --task=gem




