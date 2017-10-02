clear all
a = 0;
c = pi;
z = 0;
r = rand(1,10);
parfor i = 1:10
    a = i;
    z = z + i;
    r(i) = r(i) + 1;
end
delete(gcp)