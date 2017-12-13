kWinRad = 7;
kLumiThred = 40;
max_depth_val = 70;
min_depth_val = 20;
error_thred = 4.0;
grid_size = 15;

% Pattern: pattern.png -> pattern.txt
pat_img = imread('pattern_2size2color8P0.png');
pat_img_d = (double(pat_img) / 255) * 50 + 50;
save('result/pattern.txt', 'pat_img_d', '-ascii');

% Weight: weight.txt
weight_mat = zeros(800, 1280);
for h = 2:2:800
  for w = 2:2:1280
    weight_mat(h, w) = 1.0;
  end
end
save('result/weight.txt', 'weight_mat', '-ascii');

% Image: img.png
img_obs = imread('dyna_mat3.png');
img_obs_d = double(img_obs);
imwrite(img_obs, 'result/img.png');

% Mask: mask.txt. According to max & min val
epi_A_mat = load('EpiLine_A.txt');
epi_B_mat = load('EpiLine_B.txt');
mask_mat = double((img_obs > 15) .* (epi_A_mat ~= 0));
imshow(mask_mat);
save('result/mask.txt', 'mask_mat', '-ascii');

% EpiA, EpiB (pass)

% Mat M & D
cam_mat_0 = [2428.270501026523, 0, 717.1879617522386;
    0, 2425.524847530806, 419.6450731465209;
    0, 0, 1 ];
cam_mat_1 = [2432.058972474525, 0, 762.2933947666461;
    0, 2435.900798664577, 353.2790048217345;
    0, 0, 1];
rot_mat = [0.9991682873520409, 0.01604901003987891, 0.03748550155365887;
 -0.01624095229582852, 0.9998564818205395, 0.004821538134006965;
 -0.0374027407887994, -0.00542632824227644, 0.9992855397449185];
trans_vec = [-4.672867184359712;
 0.08985783911144951;
 -1.53686618071908];
cam_0_matrix = [cam_mat_0, zeros(3, 1)];
cam_1_matrix = cam_mat_1 * [(rot_mat)', -trans_vec];
mat_M = zeros(3, 1024*1280);
mat_D = zeros(3, 1024*1280);
dx = cam_mat_0(1, 3); fx = cam_mat_0(1, 1);
dy = cam_mat_0(2, 3); fy = cam_mat_0(2, 2);
for h = 1:1024
  for w = 1:1280
    idx_k = (h-1)*1280 + w;
    tmp_vec = [(w - 1 - dx) / fx; (h - 1 - dy) / fy; 1.0];
    mat_M(:, idx_k) = cam_1_matrix(:, 1:3) * tmp_vec;
    mat_D(:, idx_k) = cam_1_matrix(:, 4);
  end
end
save('result/Mat_M.txt', 'mat_M', '-ascii');
save('result/Mat_D.txt', 'mat_D', '-ascii');

% Depth: from last depth_val
x_pro_mat = load('xpro_mat0.txt');
last_depth_mat = zeros(1024, 1280);
for h = 1:1024
  for w = 1:1280
    if x_pro_mat(h, w) > 0.0
      idx_k = (h-1)*1280 + w;
      M = mat_M(:, idx_k);
      D = mat_D(:, idx_k);
      last_depth_mat(h, w) = -(D(1)-D(3)*x_pro_mat(h,w)) / (M(1)-M(3)*x_pro_mat(h,w));
    end
  end
end
now_depth_mat = zeros(1024, 1280);
for h = 1:1024
  for w = 1:1280
    if mask_mat(h, w) < 1
      continue;
    end
    if (last_depth_mat(h, w) > min_depth_val) && (last_depth_mat(h,w) < max_depth_val)
      now_depth_mat(h, w) = last_depth_mat(h, w);
    else
      now_depth_mat(h, w) = -1;
    end
  end
end
queue = zeros(1024*1280, 3);
start_queue = 1;
end_queue = 1;
for h = 2:1023
  for w = 2:1279
    if now_depth_mat(h, w) == -1
      h_n = 0; w_n = 0;
      if now_depth_mat(h-1, w) > 0
        h_n = h-1; w_n = w;
      else
        if now_depth_mat(h, w-1) > 0
          h_n = h; w_n = w-1;
        else
          if now_depth_mat(h+1, w) > 0
            h_n = h+1; w_n = w;
          else
            if now_depth_mat(h, w+1) > 0
              h_n = h; w_n = w+1;
            end
          end
        end
      end
      if h_n ~= 0 && w_n ~= 0
        queue(end_queue, :) = [h, w, now_depth_mat(h_n, w_n)];
        now_depth_mat(h, w) = 0;
        end_queue = end_queue + 1;
      end
    end
  end
end
while start_queue < end_queue
  h = queue(start_queue, 1);
  w = queue(start_queue, 2);
  val = queue(start_queue, 3);
  now_depth_mat(h, w) = val;
  start_queue = start_queue + 1;
  if now_depth_mat(h-1, w) < 0
    queue(end_queue, :) = [h-1, w, now_depth_mat(h, w)];
    now_depth_mat(h-1, w) = 0;
    end_queue = end_queue + 1;
  end
  if now_depth_mat(h, w-1) < 0
    queue(end_queue, :) = [h, w-1, now_depth_mat(h, w)];
    now_depth_mat(h, w-1) = 0;
    end_queue = end_queue + 1;
  end
  if now_depth_mat(h+1, w) < 0
    queue(end_queue, :) = [h+1, w, now_depth_mat(h, w)];
    now_depth_mat(h+1, w) = 0;
    end_queue = end_queue + 1;
  end
  if now_depth_mat(h, w+1) < 0
    queue(end_queue, :) = [h, w+1, now_depth_mat(h, w)];
    now_depth_mat(h, w+1) = 0;
    end_queue = end_queue + 1;
  end
end
% For control point
tmp_depth_mat = zeros(1024, 1280);
for h = 1+grid_size:1024-grid_size
  for w = 1+grid_size:1280-grid_size
    if mod(h, grid_size) == 1 && mod(w, grid_size) == 1
      if now_depth_mat(h, w) == 0
        % Up,Left,Down,Right
        if now_depth_mat(h-grid_size, w) > 0
          tmp_depth_mat(h, w) = now_depth_mat(h-grid_size, w);
          continue;
        end
        if now_depth_mat(h, w-grid_size) > 0
          tmp_depth_mat(h, w) = now_depth_mat(h, w-grid_size);
          continue;
        end
        if now_depth_mat(h+grid_size, w) > 0
          tmp_depth_mat(h, w) = now_depth_mat(h+grid_size, w);
          continue;
        end
        if now_depth_mat(h, w+grid_size) > 0
          tmp_depth_mat(h, w) = now_depth_mat(h, w+grid_size);
          continue;
        end
        % UL, DL, DR, UR
        if now_depth_mat(h-grid_size, w-grid_size) > 0
          tmp_depth_mat(h, w) = now_depth_mat(h-grid_size, w-grid_size);
          continue;
        end
        if now_depth_mat(h+grid_size, w-grid_size) > 0
          tmp_depth_mat(h, w) = now_depth_mat(h+grid_size, w-grid_size);
          continue;
        end
        if now_depth_mat(h+grid_size, w+grid_size) > 0
          tmp_depth_mat(h, w) = now_depth_mat(h+grid_size, w+grid_size);
          continue;
        end
        if now_depth_mat(h+grid_size, w-grid_size) > 0
          tmp_depth_mat(h, w) = now_depth_mat(h+grid_size, w-grid_size);
          continue;
        end
      end
    end
  end
end
now_depth_mat = now_depth_mat + tmp_depth_mat;
save('result/depth.txt', 'now_depth_mat', '-ascii');