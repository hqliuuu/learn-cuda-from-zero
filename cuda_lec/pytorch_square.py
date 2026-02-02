import torch

a = torch.tensor([1., 2., 3.])

print(torch.square(a))

print(a * a)
print(a ** 2)


def time_pytorch_time(func, input):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for _ in range(5):
        func(input)

    start.record()
    func(input)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)


b = torch.rand(10000, 10000).cuda()


def square_2(a):
    return a * a


def square_3(a):
    return a ** 2


time_pytorch_time(torch.square, b)
time_pytorch_time(square_2, b)
time_pytorch_time(square_3, b)

print("=============")
print("Profiling torch.square")
print("=============")

with torch.profiler.profile() as prof:
    torch.square(b)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

print("=============")
print("Profiling a * a")
print("=============")

with torch.profiler.profile() as prof:
    square_2(b)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

print("=============")
print("Profiling a ** 2")
print("=============")

with torch.profiler.profile() as prof:
    square_3(b)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
