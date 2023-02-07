# A script which exports Zalando's mnist-like dataset
# into a mat-files.
using ImageView
using MLDatasets: FashionMNIST
using MAT

D=Dict();
for j=0:9
    local Q
    @show j
    # Export training data
    II=findall(FashionMNIST(split=:train).targets .== j);
    Q=FashionMNIST.traintensor(II)
    A=Float64.(reshape(Q,size(Q,1)^2,size(Q,3)));
    D["item$j"]=A;
end

matwrite("zalando_items.mat",D);
