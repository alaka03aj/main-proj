import os
import subprocess
import re
import matplotlib.pyplot as plt
from gradio_client import Client, handle_file

class StampedeAnalyzer:
    def __init__(self, preprocessed_dir, yolo_path, output_dir):
        self.preprocessed_dir = preprocessed_dir
        self.yolo_path = yolo_path
        self.output_dir = output_dir
        
        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'yolo_results'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'crowd_count_results'), exist_ok=True)

        # Initialize Gradio client for crowd counting
        self.crowd_count_client = Client("a-dtya/Mask-R-CNN-example")

    def run_yolo_detection(self):
        """Run YOLOv5 detection on preprocessed frames"""
        yolo_counts = []
        
        # Iterate through preprocessed frames
        for frame_file in sorted(os.listdir(self.preprocessed_dir)):
            if frame_file.endswith('.jpg'):
                frame_path = os.path.join(self.preprocessed_dir, frame_file)
                
                # Construct YOLO detection command
                yolo_cmd = [
                    'python', 
                    os.path.join(self.yolo_path, 'detect.py'),
                    '--weights', os.path.join(self.yolo_path, 'weights/yolov5s.pt'),
                    '--source', frame_path,
                    '--img-size', '1280',
                    '--conf-thres', '0.15',
                    '--iou-thres', '0.45'
                ]
                
                try:
                    # Run YOLO detection and capture output
                    result = subprocess.run(yolo_cmd, capture_output=True, text=True)
                    
                    # Extract people count using regex
                    count_match = re.search(r'People Count: (\d+)', result.stdout)
                    if count_match:
                        count = int(count_match.group(1))
                        yolo_counts.append(count)
                        print(f"Processed {frame_file}: {count} people detected")
                    else:
                        print(f"No people count found for {frame_file}")
                        yolo_counts.append(0)
                
                except Exception as e:
                    print(f"Error processing {frame_file}: {e}")
                    yolo_counts.append(0)
        
        return yolo_counts

    def run_crowd_counting(self):
        """Run crowd counting on preprocessed frames using Gradio API"""
        crowd_counts = []
        
        # Iterate through preprocessed frames
        for frame_file in sorted(os.listdir(self.preprocessed_dir)):
            if frame_file.endswith('.jpg'):
                frame_path = os.path.join(self.preprocessed_dir, frame_file)
                
                try:
                    # Use Gradio client to process the image
                    result = self.crowd_count_client.predict(
                        image=handle_file(frame_path),
                        api_name="/predict"
                    )
                    
                    # Extract crowd count from the result
                    # Assumes the result is a string like "Crowd count: X"
                    count = int(result.split(':')[-1].strip())
                    crowd_counts.append(count)
                    
                    # Save individual count to a file
                    count_file_path = os.path.join(
                        self.output_dir, 
                        'crowd_count_results', 
                        f'{frame_file}_count.txt'
                    )
                    with open(count_file_path, 'w') as f:
                        f.write(f"Crowd Count: {count}")
                    
                    print(f"Processed {frame_file}: {count} objects detected")
                
                except Exception as e:
                    print(f"Error processing {frame_file}: {e}")
                    crowd_counts.append(0)
        
        # Create a summary file
        summary_path = os.path.join(
            self.output_dir, 
            'crowd_count_results', 
            'crowd_count_summary.txt'
        )
        with open(summary_path, 'w') as f:
            for frame_file, count in zip(
                sorted(os.listdir(self.preprocessed_dir)), 
                crowd_counts
            ):
                if frame_file.endswith('.jpg'):
                    f.write(f"{frame_file}: {count}\n")
        
        return crowd_counts

    def generate_comparison_graph(self, yolo_counts, crowd_counts):
        """Generate and save comparison graph"""
        plt.figure(figsize=(12, 6))
        plt.plot(yolo_counts, label='YOLO Object Detection', marker='o')
        plt.plot(crowd_counts, label='Crowd Counting', marker='x')
        plt.title('Crowd Analysis Comparison')
        plt.xlabel('Frame Number')
        plt.ylabel('Count')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # Save the graph
        graph_path = os.path.join(self.output_dir, 'crowd_analysis_comparison.png')
        plt.savefig(graph_path)
        plt.close()

    # def save_counts_to_file(self, yolo_counts, crowd_counts):
    #     """Save counts to a text file"""
    #     counts_file = os.path.join(self.output_dir, 'crowd_analysis_counts.txt')
    #     with open(counts_file, 'w') as f:
    #         f.write("Frame\tYOLO Count\tCrowd Count\n")
    #         for i, (yolo, crowd) in enumerate(zip(yolo_counts, crowd_counts)):
    #             f.write(f"{i}\t{yolo}\t{crowd}\n")

    def save_counts_to_file(self, yolo_counts, crowd_counts):
        """Save counts to a CSV file, incorporating pre-existing Target values"""
        # Path to the original labels.csv
        original_csv = os.path.join(self.output_dir, 'labels.csv')
        # Path to the new labels.csv
        new_csv = os.path.join(self.output_dir, 'labels_new.csv')
        
        # Dictionary to store the original Target values
        target_values = {}
        
        # Read the original labels.csv
        with open(original_csv, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                target_values[row['Image']] = row['Target']
        
        # Write to the new CSV file
        with open(new_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Write the header
            writer.writerow(['Image', 'yolo', 'mrcnn', 'target'])
            
            # Write the data
            for i, (yolo, crowd) in enumerate(zip(yolo_counts, crowd_counts)):
                # Assuming the image names are in the format '1.jpg', '2.jpg', etc.
                image_name = f'{i+1}.jpg'
                # Retrieve the target value from the dictionary
                target = target_values.get(image_name, 'Unknown')  # Default to 'Unknown' if not found
                writer.writerow([image_name, yolo, crowd, target])

    def analyze(self):
        """Main analysis pipeline"""
        # Run YOLO detection
        yolo_counts = self.run_yolo_detection()
        
        # Run crowd counting
        crowd_counts = self.run_crowd_counting()
        
        # Generate comparison graph
        self.generate_comparison_graph(yolo_counts, crowd_counts)
        
        # Save counts to file
        self.save_counts_to_file(yolo_counts, crowd_counts)
        
        return yolo_counts, crowd_counts

def main():
    # Paths - adjust these to your specific setup
    # preprocessed_dir = 'output_preprocessed/preprocessed'
    preprocessed_dir = 'images'
    yolo_path = '../YOLO-adi'
    output_dir = 'output/analysis'

    # Create analyzer
    analyzer = StampedeAnalyzer(
        preprocessed_dir, 
        yolo_path, 
        output_dir
    )

    # Run analysis
    yolo_counts, crowd_counts = analyzer.analyze()
    os.rename(os.path.join(output_dir, 'labels_new.csv'), os.path.join(output_dir, 'labels.csv'))


if __name__ == "__main__":
    main()