- #### To improve transparency and reproducibility, we provide an *overview table* (Table: dataset_overview_schema) summarizing the role of each dataset CSV file, their key columns, and their purposes. This table gives a quick reference of how the dataset is organized.

- #### Then, we offer a detailed description of each file (Table: image_dataset_schema, Table: segment_profile_schema, Table: image_to_trajectory_schema) to ensure clarity and reproducibility.

- #### Based on the above schema, we also provide a dataset sample that includes images, textual labels, and their corresponding trajectories and road network attributes.




# Table I: Overview of dataset schema across different CSV files

| **CSV File** | **Key Columns** | **Description** |
|---|---|---|
| `image_dataset.csv` | `image_path`<br>`label_name_list` *(list of strings)*<br>`label_index_list` *(list of ints)* | Full dataset of traffic scene images. Each row links an image to its multi-aspect labels (e.g., scene, surface, width, accessibility). The label information is stored as lists. |
| `segment_profile.csv` | `segment_id`<br>`function_class`<br>`lane_number`<br>`speed_class`<br>`road_length`<br>`out_degree`<br>`trajectory_count`<br>`medium_speed`<br>`other_attrs_json` | Road segment profile table. Includes both static attributes (e.g., function class, lane number, road length) and dynamic statistics (e.g., trajectory count, median speed). The JSON field preserves additional extensible attributes. |
| `image_to_trajectory.csv` | `image_path`<br>`trajectory_segments` *(list of segment IDs; semicolon-separated in CSV)*<br>`image_to_segment` | Mapping table between images and their road-segment trajectories. Each image is linked to a sequence of road segments (trajectory), and the specific segment where the image was captured is also provided. For CSV storage, the segment list is represented as a semicolon-separated string. |

# Table II: Schema of `image_dataset.csv`

| **Column**         | **Type**       | **Description**                        | **Example**                                     |
|--------------------|----------------|----------------------------------------|-------------------------------------------------|
| `image_path`       | string         | Path to the traffic scene image.        | path/to/image.jpg                               |
| `label_name_list`  | list(string)   | List of the class label name.           | ["vehicles", "normal", "normal", "hard"]        |
| `label_index_list` | list(int)      | List of the class label index.          | [1, 0, 0, 1]                                    |


# Table III: Schema of `segment_profile.csv`

| **Column**          | **Type** | **Description**                                             | **Example** |
|---------------------|----------|-------------------------------------------------------------|-------------|
| `segment_id`        | int      | Unique identifier of the road segment.                      | 0           |
| `function_class`    | int      | Degree of the segment function.                             | 4           |
| `lane_number`       | int      | Number of lanes.                                            | 2           |
| `speed_class`       | int      | Speed limit in km/h.                                        | 60          |
| `road_length`       | float    | Segment length in meters.                                   | 114.6       |
| `out_degree`        | int      | Number of downstream segments.                              | 3           |
| `trajectory_count`  | int      | Number of passing vehicles in a time window.                | 23          |
| `medium_speed`      | float    | Medium passing speed of passing vehicles in a time window.  | 34.3        |
| `other_attrs_json`  | string   | JSON dump of remaining attributes (optional, extensibility).| /           |


# Table IV: Schema of `image_to_trajectory.csv`

| **Column**            | **Type**    | **Description**                           | **Example**                         |
|-----------------------|-------------|-------------------------------------------|-------------------------------------|
| `image_path`          | string      | Path to the traffic scene image.           | path/to/image.jpg                   |
| `trajectory_segments` | list(int)   | List of segment IDs along the trajectory.  | [4323, 3451, 3361, 3312, 2453]      |
| `image_to_segment`    | int         | Segment ID where the image is captured.    | 3361                                |
