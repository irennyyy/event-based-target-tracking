import csv
from pathlib import Path
from tqdm import tqdm
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore, get_types_from_msg

# ===== Path Configuration =====
BAG_PATH = Path('/Data/shapes_rotation/raw/shapes_rotation.bag')
MSG_DIR  = Path('/Users/irenejiang/Desktop/24fall/individual project/workdesk/dvs_msgs/msg')   # contains Event.msg / EventArray.msg
TOPIC    = '/dvs/events'

# Output: Data/shapes_rotation/events_clean.csv
OUTPUT_DIR  = Path('/Data/shapes_rotation')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_CSV  = OUTPUT_DIR / 'events_clean.csv'

def ts_to_us(ts) -> int:
    """Convert to microseconds; compatible with sec/secs and nsec/nsecs/nanosec naming."""
    sec  = getattr(ts, 'sec',  getattr(ts, 'secs',  0))
    nsec = getattr(ts, 'nanosec', getattr(ts, 'nsec', getattr(ts, 'nsecs', 0)))
    return int(sec) * 1_000_000 + (int(nsec) // 1000)

def register_dvs_msgs(typestore, msg_dir: Path):
    """Parse and register custom DVS messages using get_types_from_msg."""
    event_path = msg_dir / 'Event.msg'
    array_path = msg_dir / 'EventArray.msg'
    if not event_path.exists() or not array_path.exists():
        raise FileNotFoundError(f'Event.msg or EventArray.msg not found in directory: {msg_dir}')

    add_types = {}
    add_types.update(get_types_from_msg(event_path.read_text(), 'dvs_msgs/msg/Event'))
    add_types.update(get_types_from_msg(array_path.read_text(), 'dvs_msgs/msg/EventArray'))
    typestore.register(add_types)

def convert_ros1_bag_to_csv(bag_path: Path, output_csv: Path, msg_dir: Path, topic: str = TOPIC):
    if not bag_path.exists():
        raise FileNotFoundError(f"Bag file not found: {bag_path}")
    if not msg_dir.exists():
        raise FileNotFoundError(f"Message directory not found: {msg_dir}")

    print(f"[INFO] Reading ROS1 bag file: {bag_path}")
    print(f"[INFO] Using custom message directory: {msg_dir}")

    typestore = get_typestore(Stores.ROS1_NOETIC)
    register_dvs_msgs(typestore, msg_dir)

    # Count number of events
    total_events = 0
    any_topic_found = False
    with Reader(bag_path) as reader:
        for connection, _, raw in reader.messages():
            if connection.topic == topic:
                any_topic_found = True
                msg = typestore.deserialize_ros1(raw, connection.msgtype)  # ← correct API for ROS1
                total_events += len(msg.events)
    if not any_topic_found:
        raise RuntimeError(f"Topic not found in bag: {topic}")

    print(f"[INFO] Total events: {total_events:,}")

    # Write CSV (four columns)
    with Reader(bag_path) as reader, open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp_us', 'x', 'y', 'polarity'])
        with tqdm(total=total_events, desc="Parsing events", unit="evt") as pbar:
            for connection, _, raw in reader.messages():
                if connection.topic == topic:
                    msg = typestore.deserialize_ros1(raw, connection.msgtype)
                    for e in msg.events:
                        writer.writerow([ts_to_us(e.ts), int(e.x), int(e.y), int(e.polarity)])
                        pbar.update(1)

    print(f"[INFO] CSV generated: {output_csv}")

if __name__ == "__main__":
    convert_ros1_bag_to_csv(BAG_PATH, OUTPUT_CSV, MSG_DIR)
