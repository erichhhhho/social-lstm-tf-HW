import numpy as np
import tensorflow as tf

import os
import pickle
import argparse
import ipdb

from social_utils import SocialDataLoader
from social_model import SocialModel
from grid import getSequenceGridMask
# from social_train import getSocialGrid, getSocialTensor


def get_mean_error(predicted_traj, true_traj, observed_length, maxNumPeds):
    '''
    Function that computes the mean euclidean distance error between the
    predicted and the true trajectory
    params:
    predicted_traj : numpy matrix with the points of the predicted trajectory
    true_traj : numpy matrix with the points of the true trajectory
    observed_length : The length of trajectory observed
    '''
    # The data structure to store all errors
    error = np.zeros(len(true_traj) - observed_length)
    # For each point in the predicted part of the trajectory
    for i in range(observed_length, len(true_traj)):
        # The predicted position. This will be a maxNumPeds x 3 matrix
        pred_pos = predicted_traj[i, :]
        # The true position. This will be a maxNumPeds x 3 matrix
        true_pos = true_traj[i, :]
        timestep_error = 0
        counter = 0
        for j in range(maxNumPeds):
            if true_pos[j, 0] == 0:
                # Non-existent ped
                continue
            elif pred_pos[j, 0] == 0:
                # Ped comes in the prediction time. Not seen in observed part
                continue
            else:
                if true_pos[j, 1] > 1 or true_pos[j, 1] < 0:
                    continue
                elif true_pos[j, 2] > 1 or true_pos[j, 2] < 0:
                    continue

                timestep_error += np.linalg.norm(true_pos[j, [1, 2]] - pred_pos[j, [1, 2]])
                counter += 1

        if counter != 0:
            error[i - observed_length] = timestep_error / counter

        # The euclidean distance is the error
        # error[i-observed_length] = np.linalg.norm(true_pos - pred_pos)

    # Return the mean error
    return np.mean(error)


def main():

    # Set random seed
    np.random.seed(1)

    parser = argparse.ArgumentParser()
    # Observed length of the trajectory parameter
    parser.add_argument('--obs_length', type=int, default=6,
                        help='Observed length of the trajectory')
    # Predicted length of the trajectory parameter
    parser.add_argument('--pred_length', type=int, default=6,
                        help='Predicted length of the trajectory')
    # Test dataset
    parser.add_argument('--test_dataset', type=int, default=1,
                        help='Dataset to be tested on')

    # Model to be loaded
    parser.add_argument('--epoch', type=int, default=39,
                        help='Epoch of model to be loaded')
    

    # Parse the parameters
    sample_args = parser.parse_args()

    # Save directory
    '''Original Setting'''
    # save_directory = '/home/hesl/PycharmProjects/social-lstm-tf-HW/ResultofTrainingVelocity1only/save'

    '''KITTI Training Setting'''
    save_directory = '/home/hesl/PycharmProjects/social-lstm-tf-HW/save'

    # Define the path for the config file for saved args
    with open(os.path.join(save_directory, 'social_config.pkl'), 'rb') as f:
        saved_args = pickle.load(f)

    # Create a SocialModel object with the saved_args and infer set to true
    model = SocialModel(saved_args, True)
    # Initialize a TensorFlow session
    sess = tf.InteractiveSession()
    # Initialize a saver
    saver = tf.train.Saver()

    # Get the checkpoint state for the model
    ckpt = tf.train.get_checkpoint_state(save_directory)
    # print ('loading model: ', ckpt.model_checkpoint_path)
    print('loading model: ', ckpt.model_checkpoint_path)

    # Restore the model at the checkpoint
    saver.restore(sess, ckpt.model_checkpoint_path)

    # Dataset to get data from
    dataset = [sample_args.test_dataset]

    # Create a SocialDataLoader object with batch_size 1 and seq_length equal to observed_length + pred_length
    data_loader = SocialDataLoader(1, sample_args.pred_length + sample_args.obs_length, saved_args.maxNumPeds, dataset, True, infer=True)

    # Reset all pointers of the data_loader
    data_loader.reset_batch_pointer()

    results = []

    # Variable to maintain total error
    total_error = 0
    # For each batch
    for b in range(data_loader.num_batches):
        # Get the source, target and dataset data for the next batch
        x, y, d = data_loader.next_batch(randomUpdate=False)

        # Batch size is 1
        x_batch, y_batch, d_batch = x[0], y[0], d[0]

        '''Original Setting'''
        if d_batch == 0 and dataset[0] == 0:
            dimensions = [640, 480]
        else:
            dimensions = [720, 576]

        '''eth/eth grid'''
        dimensions = [14, 16]
        '''eth/hotel grid'''
        dimensions = [7, 16]

        '''KITTI Training Setting'''
        # if d_batch == 0 and dataset[0] == 0:
        #     dimensions = [1242, 375]
        # else:
        #     dimensions = [1224, 370]
        #
        # dimensions = [1224, 370]


        grid_batch = getSequenceGridMask(x_batch, dimensions, saved_args.neighborhood_size, saved_args.grid_size)

        obs_traj = x_batch[:sample_args.obs_length]
        obs_grid = grid_batch[:sample_args.obs_length]
        # obs_traj is an array of shape obs_length x maxNumPeds x 3

        print("********************** SAMPLING A NEW TRAJECTORY", b, "******************************")
        complete_traj = model.sample(sess, obs_traj, obs_grid, dimensions, x_batch, sample_args.pred_length)

        # ipdb.set_trace()
        # complete_traj is an array of shape (obs_length+pred_length) x maxNumPeds x 3
        total_error += get_mean_error(complete_traj, x[0], sample_args.obs_length, saved_args.maxNumPeds)

        print("Processed trajectory number : ", b, "out of ", data_loader.num_batches, " trajectories")

        # plot_trajectories(x[0], complete_traj, sample_args.obs_length)
        # return
        results.append((x[0], complete_traj, sample_args.obs_length))

    # Print the mean error across all the batches
    print("Total mean error of the model is ", total_error/data_loader.num_batches)

    print("Saving results")
    with open(os.path.join(save_directory, 'social_results.pkl'), 'wb') as f:
        pickle.dump(results, f)

if __name__ == '__main__':
    main()
