import torch
import torch.nn as nn


class Tower(nn.Module):
    def __init__(self, input_dim, dims):
        super(Tower, self).__init__()
        self.input_dim = input_dim
        self.output_dim = dims[-1]

        self.tower = nn.Sequential()
        for i, dim in enumerate(dims):
            self.tower.add_module("linear_%s" % i, nn.Linear(input_dim, dim))
            self.tower.add_module("tanh_%s" % i, nn.Tanh())

    def forward(self, features):
        return self.tower(features)


def get_distance_to_ideal(query_vec, listing_vec):
    ''' Euclidean distance of the listing hidden layer to the query
    hidden layer . In practice minimizing sum of squared diﬀ is
    equivalent to minimizing the Euclidean distance . '''
    sqdiff = torch.square(query_vec - listing_vec)
    logits = sqdiff.sum(dim=-1)
    return logits


def pairwise_loss(qvec, booked_vec, not_booked_vec):
    booked_distance = get_distance_to_ideal(qvec, booked_vec)
    not_booked_distance = get_distance_to_ideal(qvec, not_booked_vec)

    loss_fn = nn.BCEWithLogitsLoss()
    loss = loss_fn(booked_distance, torch.ones_like(booked_distance)) + \
           loss_fn(not_booked_distance, torch.zeros_like(not_booked_distance))
    return loss

def main(query_features, booked_listing_features, not_booked_listing_features):
    query_tower = Tower(query_features.shape[1], [64, 32])
    list_tower = Tower(booked_listing_features.shape[1], [64, 32])

    qvec = query_tower(query_features)
    booked_vec = list_tower(booked_listing_features)
    not_booked_vec = list_tower(not_booked_listing_features)

    loss = pairwise_loss(qvec, booked_vec, not_booked_vec)