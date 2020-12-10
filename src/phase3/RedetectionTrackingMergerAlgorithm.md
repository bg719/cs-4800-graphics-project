# Redetection Tracking Merger Algorithm

The most complicated and computationally expensive part of the redetection-tracking 
sequence processing is the merging of detected and tracked object sets by associating
objects which are the same between the two. In general, the process involves:

```
1) Run tracking on current_frame

2) Run detection on current_frame

3) Calculate the bounding box centroids for all elements of the tracked objects set

4) Calculate the bounding box centroids for all elements of the detected objects set

5) Calculate the nearest neighbors for each element in the detected set from the tracked set

6) Calculate a comparison_threshold

7) For each object in the detected set:

    i) If distance_to_nearest_neigbor < comparison_threshold, associate the nearest 
    neighbor (tracked) object ID with the current object

    ii) Else, assign the object a new ID
```

However, a number of challenges present themselves in 
this regard:

1) The number of objects between the two sets may not be the same, due to objects entering
or exiting the scene during a tracking sequence

2) Objects that are very close in the scene may be mis-identified

3) The processing order of objects when there is a difference between the number of tracked
and detected objects may result in mis-association or the association of a single tracked
object with multiple detected objects

In order to overcome these challenges, the suggested algorithm for accomplishing the merger
is as follows:

```python
def redetect_and_merge_with_tracked(current_frame, multitracker, threshold_scalar, 
    reinitialization_threshold):
    
    # Update tracked object positions in the current frame
    tracked_objects = multitracker.update(current_frame)
    
    # Detect objects in the current frame
    detected_objects = detect_objects(current_frame)
    
    # Calculate centroids for bounding boxes of objects
    tracked_centroids = calc_centroids(tracked_objects)
    detected_centroids = calc_centroids(detected_objects)

    # Calculate a comparison threshold
    comparison_threshold = get_width(select_threshold_object(detected_objects)) * threshold_scalar
    
    # Find the distances to and indices of the nearest neighbors from the tracked set
    # to each element of the detected set
    distances, indices = calc_nearest_neigbors(detected_centroids, tracked_centroids)

    # Create a composite nearest neighbors data structure
    n_neighbors = []
    for i in range(len(detected_centroids)):
        n_neighbors[i] = {'index': i, 'distances': distances[i], 'indices': indices[i]}
    
    # Sort nearest neighbors based on the smallest distance to a nearest neighbor
    n_neighbors.sort(key=lambda t: t['distances'][0])
    
    # Initialize a set to tracked objects associated with a detected object
    associated_objs = set()
    i = 0
    while i < len(n_neighbors):
        j = 0
        n_distances = n_neighbors[i]['distances']
        n_indices = n_neighbors[i]['indices']
        
        # Iterate through the nearest neighbors as long as the associated distance
        # remains less than the comparison threshold
        while j < len(n_indices) and n_distances[j] < comparison_threshold:
            # If the index is not already associated with a detected object, 
            # associate it with this detected object
            if n_indices[j] not in associated_objs:
                associated_objs.add(n_indices[j])
                reassign_id(tracked_objects[n_indices[j]], detected_objects[n_neighbors[i]['index']])
            # Otherwise, iterate j to check the next nearest neighbor    
            else:
                j += 1
        
        # If we have tried all nearest neighbor indices or surpassed the 
        # comparison threshold, break out of the loop
        if j == len(n_indices) or n_distances[j] >= comparison_threshold:
            break
    
    # Check if all detected objects have been assigned an id
    if i == len(n_neighbors):
        return
    # Check if enough objects have been redetected successfully to avoid 
    # a re-initialization
    elif i/len(n_neighbors) < reinitialization_threshold:
        initialize()
        return
    
    # Assign new IDs to any remaining detected objects
    while i < len(n_neighbors):
        assign_new_id(detected_objects[i])
        i += 1
    
    return
```