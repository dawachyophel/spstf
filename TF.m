
function F = TF(img, sigma_s, sigma_r, num_iterations, joint_image)

    I = double(img);

    if ~exist('num_iterations', 'var')
        num_iterations = 3;
    end
    
    if exist('joint_image', 'var') && ~isempty(joint_image)
        J = double(joint_image);
    
        if (size(I,1) ~= size(J,1)) || (size(I,2) ~= size(J,2))
            error('Input and joint images must have equal width and height.');
        end
    else
        J = I;
    end
    
    [h w num_joint_channels] = size(J);

    % Estimate horizontal and vertical partial derivatives using finite
    % differences.
    dIcdx = diff(J, 1, 2);
    dIcdy = diff(J, 1, 1);
    
    dIdx = zeros(h,w);
    dIdy = zeros(h,w);
    
    % Compute the l1-norm distance of neighbor pixels.
    for c = 1:num_joint_channels
        dIdx(:,2:end) = dIdx(:,2:end) + abs( dIcdx(:,:,c) );
        dIdy(2:end,:) = dIdy(2:end,:) + abs( dIcdy(:,:,c) );
    end
    
    % Compute the derivatives of the horizontal and vertical domain transforms.
    dHdx = (1 + sigma_s/sigma_r * dIdx);
    dVdy = (1 + sigma_s/sigma_r * dIdy);
    
    
    % The vertical pass is performed using a transposed image.
    dVdy = dVdy';
    
    %% Perform the transform filtering.
    N = num_iterations;
    F = I;
    
    sigma_H = sigma_s;
    
    for i = 0:num_iterations - 1
    
        % Compute the sigma value for this iteration 
        sigma_H_i = sigma_H * sqrt(3) * 2^(N - (i + 1)) / sqrt(4^N - 1);
    
        F = TransformFilter_Horizontal(F, dHdx, sigma_H_i);
        F = image_transpose(F);
    
        F = TransformFilter_Horizontal(F, dVdy, sigma_H_i);
        F = image_transpose(F);
        
    end
    
    F = cast(F, class(img));

end

%% Transform filter.
function F = TransformFilter_Horizontal(I, D, sigma)

    % Feedback coefficient 
    a = exp(-sqrt(2) / sigma);
    
    F = I;
    V = a.^D;
    
    [h w num_channels] = size(I);
    
    % Left -> Right filter.
    for i = 2:w
        for c = 1:num_channels
            F(:,i,c) = F(:,i,c) + V(:,i) .* ( F(:,i - 1,c) - F(:,i,c) );
        end
    end
    
    % Right -> Left filter.
    for i = w-1:-1:1
        for c = 1:num_channels
            F(:,i,c) = F(:,i,c) + V(:,i+1) .* ( F(:,i + 1,c) - F(:,i,c) );
        end
    end

end

%%
function T = image_transpose(I)

    [h w num_channels] = size(I);
    
    T = zeros([w h num_channels], class(I));
    
    for c = 1:num_channels
        T(:,:,c) = I(:,:,c)';
    end
    
end
