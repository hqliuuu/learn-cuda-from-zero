# Reduce分析性能

## 最朴素reduce：simple_reduce.cu
每个线程获取一个元素，隔stride对两个元素逐步做加法。但是容易出现thread空闲情况。这里的Branch Efficiency达到74。线程束分化严重。
```
 Section: Source Counters
    ------------------------- ----------- ------------
    Metric Name               Metric Unit Metric Value
    ------------------------- ----------- ------------
    Branch Instructions Ratio           %         0.12
    Branch Instructions              inst        1,312
    Branch Efficiency                   %        74.05
    Avg. Divergent Branches                       2.39
    ------------------------- ----------- ------------
```
利用间域并行计算解决线程束分化问题: control_divergence.cu:
```
    Section: Source Counters
    ------------------------- ----------- ------------
    Metric Name               Metric Unit Metric Value
    ------------------------- ----------- ------------
    Branch Instructions Ratio           %         0.31
    Branch Instructions              inst        1,126
    Branch Efficiency                   %        99.32
    Avg. Divergent Branches                       0.06
    ------------------------- ----------- ------------
```
将数据搬到 shared memory : shared_reduce.cu
我这里尝试过将矩阵的size改成4096，发现最后的sum的计算结果是0，而不是4096。这是因为我是只用了一个block里的线程去reduce所有数据。而且我是将所有的数据都放在shared_memory。
我查询了下安培架构下的GPU block_size最大是1024。我认为是我设置的block_size超出了这个架构的最大限制，导致计算出来的结果出错。
````
    Section: Source Counters
    ------------------------- ----------- ------------
    Metric Name               Metric Unit Metric Value
    ------------------------- ----------- ------------
    Branch Instructions Ratio           %         0.10
    Branch Instructions              inst          385
    Branch Efficiency                   %          100
    Avg. Divergent Branches                          0
    ------------------------- ----------- ------------
````

thread coarsening 让一个线程处理更多的数据。  : reduce_coarsening.cu

但是一旦数据过大，超出共享内存，就无法计算。将数据切割到多个block中计算，最后汇总。