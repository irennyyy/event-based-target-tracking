import csv
from pathlib import Path
from tqdm import tqdm
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore, get_types_from_msg
import argparse

def ts_to_us(ts) -> int:
    """Convert ROS timestamp to microseconds."""
    sec  = getattr(ts, 'sec',  getattr(ts, 'secs',  0))
    nsec = getattr(ts, 'nanosec', getattr(ts, 'nsec', getattr(ts, 'nsecs', 0)))
    return int(sec) * 1_000_000 + (int(nsec) // 1000)

def register_dvs_msgs(typestore, msg_dir: Path):
    """Register custom DVS messages (Event, EventArray) into the type store."""
    event_path = msg_dir / 'Event.msg'
    array_path = msg_dir / 'EventArray.msg'
    if not event_path.exists() or not array_path.exists():
        raise FileNotFoundError(f'Event.msg or EventArray.msg not found in directory: {msg_dir}')

    add_types = {}
    add_types.update(get_types_from_msg(event_path.read_text(), 'dvs_msgs/msg/Event'))
    add_types.update(get_types_from_msg(array_path.read_text(), 'dvs_msgs/msg/EventArray'))
    typestore.register(add_types)

def convert_ros1_bag_to_csv(bag_path: Path, output_csv: Path, msg_dir: Path, topic: str = '/dvs/events'):
    """Convert a ROS1 bag file with DVS event messages to a clean CSV file."""
    if not bag_path.exists():
        raise FileNotFoundError(f"Bag file not found: {bag_path}")
    if not msg_dir.exists():
        raise FileNotFoundError(f"Message definition directory not found: {msg_dir}")

    print(f"[INFO] Reading ROS1 bag file: {bag_path}")
    print(f"[INFO] Using custom message definition directory: {msg_dir}")

    typestore = get_typestore(Stores.ROS1_NOETIC)
    register_dvs_msgs(typestore, msg_dir)

    # Count total number of events
    total_events = 0
    with Reader(bag_path) as reader:
        for connection, _, raw in reader.messages():
            if connection.topic == topic:
                msg = typestore.deserialize_ros1(raw, connection.msgtype)
                total_events += len(msg.events)

    print(f"[INFO] Total events: {total_events:,}")

    # Write CSV (four columns)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", type=str, required=True, help="Input ROS1 bag file path")
    parser.add_argument("--out_csv", type=str, required=True, help="Output CSV file path")
    parser.add_argument("--msg_dir", type=str, default="dvs_msgs/msg", help="Custom message directory")
    parser.add_argument("--topic", type=str, default="/dvs/events", help="Event topic name")
    args = parser.parse_args()

    convert_ros1_bag_to_csv(Path(args.bag), Path(args.out_csv), Path(args.msg_dir), args.topic)
