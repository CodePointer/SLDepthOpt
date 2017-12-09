color_set = zeros(256, 1);
for h = 1:1024
  for w = 1:1280
    if mask_mat(h, w) == 1
      color_set(img_obs(h, w)) = color_set(img_obs(h, w)) + 1;
    end
  end
end