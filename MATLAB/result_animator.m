% import data

run_num = 2051;

result = open(sprintf('mat_files/result%d.mat',run_num));
state = result.th;
t = result.tvec;
cart_properties = [50,1,1,5,5];

% generate frames
filename = sprintf("animations/cart_animation%d",run_num);
obj = VideoWriter(filename,'MPEG-4'); % Specify the file format
obj.FrameRate = 60; % Set the frame rate
open(obj);
cullrate = 5;
tmp = 1;
for i = 1:length(t)
    if(tmp == cullrate)
        draw_cart(state(i,:),cart_properties,[-12,12])
        writeVideo(obj,getframe)
        clf
        tmp = 0;
    end
    tmp = tmp + 1;
end
close(obj)
close all

% animator function
function draw_cart(state,cart_properties,bounds)
L1 = cart_properties(4);
L2 = cart_properties(5);

% plot cart
p1 = [state(1) - L1/2,L1/2];
p2 = [state(1) + L1/2,L1/2];
p3 = [state(1) + L1/2,-L1/2];
p4 = [state(1) - L1/2,-L1/2];
cart_points = vertcat(p1,p2,p3,p4,p1);
plot(cart_points(:,1),cart_points(:,2));

hold on

% other stuff
yline(0)
xline(0)

% plot pendulum 1
p6 = [state(1) + L1 * sin(state(3)),L1 * cos(state(3))];
pend_points = vertcat([state(1),0],p6,[state(1),0]);
plot(pend_points(:,1),pend_points(:,2))

% plot pendulum 2
p7 = [p6(1) + L2 * sin(state(5)),p6(2) + L2 * cos(state(5))];
pend_points = vertcat(p6,p7,p6);
plot(pend_points(:,1),pend_points(:,2))

axis equal
xlim(bounds)
ylim(bounds)
end