/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

 	default_random_engine gen;

  num_particles = 350;

	// Create normal distributions for x, y and theta.
	normal_distribution<double> N_x(x, std[0]);
	normal_distribution<double> N_y(y, std[1]);
	normal_distribution<double> N_theta(theta, std[2]);

	for (int i = 0; i < num_particles; i++){

		Particle particle;

		particle.id = i;
		particle.x = N_x(gen);
		particle.y = N_y(gen);
		particle.theta = N_theta(gen);
		particle.weight = 1;

		particles.push_back(particle);
		weights.push_back(1);

	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	// Some constants to save computation power

 	default_random_engine gen;

	for (int i = 0; i < num_particles; i++){
    if (fabs(yaw_rate) == 0){

			particles[i].x += velocity * delta_t * cos(particles[i].theta);
			particles[i].y += velocity * delta_t * sin(particles[i].theta);

    } else {

			particles[i].x += velocity/yaw_rate * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
			particles[i].y += velocity/yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + (yaw_rate * delta_t)));
			particles[i].theta += yaw_rate * delta_t;

    }

		normal_distribution<double> N_x(0.0, std_pos[0]);
		normal_distribution<double> N_y(0.0, std_pos[1]);
		normal_distribution<double> N_theta(0.0, std_pos[2]);

		// Add noise
		particles[i].x += N_x(gen);
		particles[i].y += N_y(gen);
		particles[i].theta += N_theta(gen);

	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
	const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	const double std_x = std_landmark[0];
	const double std_y = std_landmark[1];

	const double k = 2 * M_PI * std_x * std_y;

	double weights_sum = 0.0;

	for (int i = 0; i < num_particles; i++){
		double weight = 0.0;
		const double sin_theta = sin(particles[i].theta);
		const double cos_theta = cos(particles[i].theta);

		for (int j = 0; j < observations.size(); j++){
			LandmarkObs l_obs;
			l_obs.id = observations[j].id;
			l_obs.x = particles[i].x + (observations[j].x * cos_theta) - (observations[j].y * sin_theta);
			l_obs.y = particles[i].y + (observations[j].x * sin_theta) + (observations[j].y * cos_theta);

			Map::single_landmark_s nearest;
      double min_dist = 10000000.0;
			bool close = false;

      for (int k = 0; k < map_landmarks.landmark_list.size(); k++) {
        Map::single_landmark_s lm = map_landmarks.landmark_list[k];
        double distance = dist(lm.x_f, lm.y_f, l_obs.x, l_obs.y);

        if (distance < min_dist) {
          min_dist = distance;
          nearest = lm;
          if (distance < sensor_range){
						close = true;
					}
        }
      }

      if (close){
				double dx = l_obs.x - nearest.x_f;
				double dy = l_obs.y - nearest.y_f;
				weight += dx * dx / (std_x * std_x) + dy * dy / (std_y * std_y);
			} else {
				weight += 100;
			}
		}
		particles[i].weight = exp(-0.5 * weight);
		weights_sum += particles[i].weight;
	}

	// Normalize weights
	for (int i = 0; i < num_particles; i++){
		particles[i].weight /= weights_sum * k;
		weights[i] = particles[i].weight;
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

 	default_random_engine gen;

	discrete_distribution<> dp(weights.begin(), weights.end());
	vector<Particle> res;

	for (int i = 0; i < num_particles; i++) {
		res.push_back(particles[dp(gen)]);
	}

	particles = res;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations,
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
