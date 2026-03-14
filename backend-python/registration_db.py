"""
RoadVision — Mock Vehicle Registration Database
Simulates a vehicle registration lookup system for testing ANPR.

This module provides a lightweight in-memory registry of Indian vehicles
for validating detected number plates against registered vehicles.
"""

import re
from typing import Dict, Optional

# Mock Vehicle Registration Database
# Format: SSDDLLNNNN (State-District-Letters-Numbers)
MOCK_REGISTRY = {
    # Tamil Nadu (TN)
    "TN10AB1234": {
        "owner_name": "Ravi Kumar",
        "vehicle_type": "Car",
        "state": "Tamil Nadu",
        "rto": "Chennai South",
        "model": "Maruti Swift",
        "color": "White"
    },
    "TN01CD5678": {
        "owner_name": "Priya Sharma",
        "vehicle_type": "Motorcycle",
        "state": "Tamil Nadu",
        "rto": "Chennai Central",
        "model": "Honda Activa",
        "color": "Red"
    },
    "TN09EF9012": {
        "owner_name": "Suresh Babu",
        "vehicle_type": "Car",
        "state": "Tamil Nadu",
        "rto": "Coimbatore",
        "model": "Hyundai i20",
        "color": "Blue"
    },
    "TN22GH3456": {
        "owner_name": "Lakshmi Devi",
        "vehicle_type": "Scooter",
        "state": "Tamil Nadu",
        "rto": "Madurai",
        "model": "TVS Jupiter",
        "color": "Black"
    },
    "TN07IJ7890": {
        "owner_name": "Venkat Raman",
        "vehicle_type": "Commercial",
        "state": "Tamil Nadu",
        "rto": "Trichy",
        "model": "Tata Ace",
        "color": "Yellow"
    },
    "TN38KL2345": {
        "owner_name": "Anitha Krishnan",
        "vehicle_type": "Car",
        "state": "Tamil Nadu",
        "rto": "Salem",
        "model": "Honda City",
        "color": "Silver"
    },
    "TN45MN6789": {
        "owner_name": "Rajesh Kumar",
        "vehicle_type": "Motorcycle",
        "state": "Tamil Nadu",
        "rto": "Tirunelveli",
        "model": "Royal Enfield",
        "color": "Black"
    },
    
    # Karnataka (KA)
    "KA05MN7788": {
        "owner_name": "Ajay Singh",
        "vehicle_type": "Motorcycle",
        "state": "Karnataka",
        "rto": "Bangalore Central",
        "model": "Bajaj Pulsar",
        "color": "Blue"
    },
    "KA01AA4321": {
        "owner_name": "Deepa Rao",
        "vehicle_type": "Car",
        "state": "Karnataka",
        "rto": "Bangalore East",
        "model": "Toyota Innova",
        "color": "White"
    },
    "KA03BB8765": {
        "owner_name": "Karthik Gowda",
        "vehicle_type": "Car",
        "state": "Karnataka",
        "rto": "Bangalore North",
        "model": "Tata Nexon",
        "color": "Red"
    },
    "KA09CC1122": {
        "owner_name": "Manjula Reddy",
        "vehicle_type": "Scooter",
        "state": "Karnataka",
        "rto": "Mysore",
        "model": "Honda Dio",
        "color": "Pink"
    },
    "KA20DD3344": {
        "owner_name": "Sunil Kumar",
        "vehicle_type": "Commercial",
        "state": "Karnataka",
        "rto": "Hubli",
        "model": "Mahindra Bolero",
        "color": "White"
    },
    "KA51EE5566": {
        "owner_name": "Ramya Shetty",
        "vehicle_type": "Car",
        "state": "Karnataka",
        "rto": "Mangalore",
        "model": "Maruti Baleno",
        "color": "Grey"
    },
    "KA04FF7788": {
        "owner_name": "Prakash Hegde",
        "vehicle_type": "Motorcycle",
        "state": "Karnataka",
        "rto": "Bangalore South",
        "model": "KTM Duke",
        "color": "Orange"
    },
    
    # Delhi (DL)
    "DL01AA4321": {
        "owner_name": "Amit Verma",
        "vehicle_type": "Car",
        "state": "Delhi",
        "rto": "Central Delhi",
        "model": "Hyundai Creta",
        "color": "Black"
    },
    "DL3CAB1234": {
        "owner_name": "Neha Kapoor",
        "vehicle_type": "Car",
        "state": "Delhi",
        "rto": "East Delhi",
        "model": "Maruti Dzire",
        "color": "White"
    },
    "DL8SBC5678": {
        "owner_name": "Rohit Sharma",
        "vehicle_type": "Motorcycle",
        "state": "Delhi",
        "rto": "South Delhi",
        "model": "Hero Splendor",
        "color": "Black"
    },
    "DL5CAD9012": {
        "owner_name": "Pooja Malhotra",
        "vehicle_type": "Scooter",
        "state": "Delhi",
        "rto": "West Delhi",
        "model": "Suzuki Access",
        "color": "Blue"
    },
    "DL1CAE3456": {
        "owner_name": "Vikram Singh",
        "vehicle_type": "Commercial",
        "state": "Delhi",
        "rto": "North Delhi",
        "model": "Ashok Leyland",
        "color": "Yellow"
    },
    "DL7CAF7890": {
        "owner_name": "Sonia Gupta",
        "vehicle_type": "Car",
        "state": "Delhi",
        "rto": "New Delhi",
        "model": "Honda Amaze",
        "color": "Silver"
    },
    "DL2CAG2345": {
        "owner_name": "Arjun Mehta",
        "vehicle_type": "Motorcycle",
        "state": "Delhi",
        "rto": "Central Delhi",
        "model": "Yamaha FZ",
        "color": "Red"
    },
    
    # Maharashtra (MH)
    "MH12DE1433": {
        "owner_name": "Sanjay Patil",
        "vehicle_type": "Car",
        "state": "Maharashtra",
        "rto": "Pune",
        "model": "Volkswagen Polo",
        "color": "White"
    },
    "MH01AB5678": {
        "owner_name": "Priyanka Desai",
        "vehicle_type": "Car",
        "state": "Maharashtra",
        "rto": "Mumbai Central",
        "model": "Skoda Rapid",
        "color": "Grey"
    },
    "MH02CD9012": {
        "owner_name": "Rahul Joshi",
        "vehicle_type": "Motorcycle",
        "state": "Maharashtra",
        "rto": "Mumbai West",
        "model": "Bajaj Avenger",
        "color": "Black"
    },
    "MH14EF3456": {
        "owner_name": "Sneha Kulkarni",
        "vehicle_type": "Scooter",
        "state": "Maharashtra",
        "rto": "Nashik",
        "model": "TVS Scooty",
        "color": "Red"
    },
    "MH20GH7890": {
        "owner_name": "Anil Pawar",
        "vehicle_type": "Commercial",
        "state": "Maharashtra",
        "rto": "Nagpur",
        "model": "Eicher Truck",
        "color": "Blue"
    },
    "MH04IJ2345": {
        "owner_name": "Kavita Shah",
        "vehicle_type": "Car",
        "state": "Maharashtra",
        "rto": "Thane",
        "model": "Renault Kwid",
        "color": "Orange"
    },
    "MH43KL6789": {
        "owner_name": "Ganesh Rane",
        "vehicle_type": "Motorcycle",
        "state": "Maharashtra",
        "rto": "Aurangabad",
        "model": "Honda CB Shine",
        "color": "Blue"
    },
    
    # Kerala (KL)
    "KL01MN1234": {
        "owner_name": "Suresh Menon",
        "vehicle_type": "Car",
        "state": "Kerala",
        "rto": "Thiruvananthapuram",
        "model": "Maruti Ertiga",
        "color": "White"
    },
    "KL07OP5678": {
        "owner_name": "Divya Nair",
        "vehicle_type": "Scooter",
        "state": "Kerala",
        "rto": "Kochi",
        "model": "Honda Activa",
        "color": "Grey"
    },
    "KL10QR9012": {
        "owner_name": "Arun Kumar",
        "vehicle_type": "Motorcycle",
        "state": "Kerala",
        "rto": "Kozhikode",
        "model": "Yamaha R15",
        "color": "Blue"
    },
    "KL14ST3456": {
        "owner_name": "Reshma Thomas",
        "vehicle_type": "Car",
        "state": "Kerala",
        "rto": "Thrissur",
        "model": "Hyundai Venue",
        "color": "Red"
    },
    "KL41UV7890": {
        "owner_name": "Vinod Pillai",
        "vehicle_type": "Commercial",
        "state": "Kerala",
        "rto": "Kollam",
        "model": "Force Traveller",
        "color": "White"
    },
    "KL03WX2345": {
        "owner_name": "Anjali Krishnan",
        "vehicle_type": "Car",
        "state": "Kerala",
        "rto": "Alappuzha",
        "model": "Ford EcoSport",
        "color": "Black"
    },
    "KL08YZ6789": {
        "owner_name": "Rajesh Varma",
        "vehicle_type": "Motorcycle",
        "state": "Kerala",
        "rto": "Kannur",
        "model": "Suzuki Gixxer",
        "color": "Red"
    },
    
    # Uttar Pradesh (UP)
    "UP16AB1234": {
        "owner_name": "Ramesh Yadav",
        "vehicle_type": "Car",
        "state": "Uttar Pradesh",
        "rto": "Noida",
        "model": "Maruti Ciaz",
        "color": "Silver"
    },
    "UP32CD5678": {
        "owner_name": "Sunita Devi",
        "vehicle_type": "Scooter",
        "state": "Uttar Pradesh",
        "rto": "Lucknow",
        "model": "TVS Wego",
        "color": "Pink"
    },
    "UP70EF9012": {
        "owner_name": "Manoj Kumar",
        "vehicle_type": "Motorcycle",
        "state": "Uttar Pradesh",
        "rto": "Kanpur",
        "model": "Hero Passion",
        "color": "Black"
    },
    "UP14GH3456": {
        "owner_name": "Asha Singh",
        "vehicle_type": "Car",
        "state": "Uttar Pradesh",
        "rto": "Ghaziabad",
        "model": "Tata Tiago",
        "color": "Blue"
    },
    "UP80IJ7890": {
        "owner_name": "Rajendra Prasad",
        "vehicle_type": "Commercial",
        "state": "Uttar Pradesh",
        "rto": "Varanasi",
        "model": "Mahindra Pickup",
        "color": "White"
    },
    "UP65KL2345": {
        "owner_name": "Geeta Sharma",
        "vehicle_type": "Car",
        "state": "Uttar Pradesh",
        "rto": "Agra",
        "model": "Nissan Magnite",
        "color": "Orange"
    },
    "UP78MN6789": {
        "owner_name": "Santosh Gupta",
        "vehicle_type": "Motorcycle",
        "state": "Uttar Pradesh",
        "rto": "Meerut",
        "model": "TVS Apache",
        "color": "Red"
    },
    
    # User Requested Plates
    "TN57AD3604": {
        "owner_name": "Arjun Prakash",
        "vehicle_type": "Car",
        "state": "Tamil Nadu",
        "rto": "Madurai South",
        "model": "Honda City",
        "color": "Silver"
    },
    "HR26CA5678": {
        "owner_name": "Vikram Singh",
        "vehicle_type": "Car",
        "state": "Haryana",
        "rto": "Faridabad",
        "model": "Hyundai Verna",
        "color": "White"
    },
    "WB02CD3456": {
        "owner_name": "Sourav Chatterjee",
        "vehicle_type": "Car",
        "state": "West Bengal",
        "rto": "Kolkata South",
        "model": "Maruti Brezza",
        "color": "Red"
    },
    
    # Additional Mixed Entries
    "TN51OP1111": {
        "owner_name": "Karthik Subramanian",
        "vehicle_type": "Car",
        "state": "Tamil Nadu",
        "rto": "Vellore",
        "model": "Kia Seltos",
        "color": "White"
    },
    "KA21QR2222": {
        "owner_name": "Meera Bhat",
        "vehicle_type": "Scooter",
        "state": "Karnataka",
        "rto": "Belgaum",
        "model": "Suzuki Burgman",
        "color": "Grey"
    },
    "DL4CST3333": {
        "owner_name": "Kunal Khanna",
        "vehicle_type": "Car",
        "state": "Delhi",
        "rto": "Dwarka",
        "model": "MG Hector",
        "color": "Black"
    },
    "MH05UV4444": {
        "owner_name": "Vaishali Bhosale",
        "vehicle_type": "Motorcycle",
        "state": "Maharashtra",
        "rto": "Kalyan",
        "model": "Honda Hornet",
        "color": "Red"
    },
    "KL13WX5555": {
        "owner_name": "Biju George",
        "vehicle_type": "Car",
        "state": "Kerala",
        "rto": "Palakkad",
        "model": "Mahindra XUV300",
        "color": "Blue"
    },
    "UP20YZ6666": {
        "owner_name": "Pankaj Mishra",
        "vehicle_type": "Commercial",
        "state": "Uttar Pradesh",
        "rto": "Allahabad",
        "model": "Tata 407",
        "color": "Yellow"
    },
    "TN99AB7777": {
        "owner_name": "Nandini Iyer",
        "vehicle_type": "Car",
        "state": "Tamil Nadu",
        "rto": "Kanchipuram",
        "model": "Jeep Compass",
        "color": "White"
    },
    "KA02CD8888": {
        "owner_name": "Harish Naik",
        "vehicle_type": "Motorcycle",
        "state": "Karnataka",
        "rto": "Bangalore West",
        "model": "Kawasaki Ninja",
        "color": "Green"
    },
    
    # Additional Plates - More Coverage
    "TN33EF4567": {
        "owner_name": "Meena Sundaram",
        "vehicle_type": "Scooter",
        "state": "Tamil Nadu",
        "rto": "Dharmapuri",
        "model": "Honda Activa 6G",
        "color": "Grey"
    },
    "TN66GH7890": {
        "owner_name": "Balaji Raman",
        "vehicle_type": "Car",
        "state": "Tamil Nadu",
        "rto": "Tiruppur",
        "model": "Tata Altroz",
        "color": "Blue"
    },
    "KA19IJ2345": {
        "owner_name": "Priya Shetty",
        "vehicle_type": "Car",
        "state": "Karnataka",
        "rto": "Shimoga",
        "model": "Maruti WagonR",
        "color": "White"
    },
    "KA32KL6789": {
        "owner_name": "Naveen Kumar",
        "vehicle_type": "Motorcycle",
        "state": "Karnataka",
        "rto": "Davangere",
        "model": "TVS Apache RTR",
        "color": "Red"
    },
    "DL5CAH1234": {
        "owner_name": "Ritu Sharma",
        "vehicle_type": "Car",
        "state": "Delhi",
        "rto": "West Delhi",
        "model": "Hyundai i10",
        "color": "Silver"
    },
    "DL9CAI5678": {
        "owner_name": "Karan Malhotra",
        "vehicle_type": "Motorcycle",
        "state": "Delhi",
        "rto": "South Delhi",
        "model": "Royal Enfield Classic",
        "color": "Black"
    },
    "MH46MN9012": {
        "owner_name": "Ashok Deshmukh",
        "vehicle_type": "Car",
        "state": "Maharashtra",
        "rto": "Ratnagiri",
        "model": "Mahindra Scorpio",
        "color": "White"
    },
    "MH15OP3456": {
        "owner_name": "Swati Kulkarni",
        "vehicle_type": "Scooter",
        "state": "Maharashtra",
        "rto": "Nashik",
        "model": "Suzuki Access 125",
        "color": "Blue"
    },
    "KL05QR7890": {
        "owner_name": "Arun Nair",
        "vehicle_type": "Car",
        "state": "Kerala",
        "rto": "Ernakulam",
        "model": "Ford Figo",
        "color": "Red"
    },
    "KL42ST2345": {
        "owner_name": "Deepa Thomas",
        "vehicle_type": "Motorcycle",
        "state": "Kerala",
        "rto": "Kasaragod",
        "model": "Honda CB Unicorn",
        "color": "Black"
    },
    "UP32UV6789": {
        "owner_name": "Amit Kumar",
        "vehicle_type": "Car",
        "state": "Uttar Pradesh",
        "rto": "Lucknow",
        "model": "Volkswagen Vento",
        "color": "Grey"
    },
    "UP75WX1234": {
        "owner_name": "Priyanka Singh",
        "vehicle_type": "Scooter",
        "state": "Uttar Pradesh",
        "rto": "Gorakhpur",
        "model": "TVS Ntorq",
        "color": "Yellow"
    },
    "HR55YZ5678": {
        "owner_name": "Rajesh Yadav",
        "vehicle_type": "Car",
        "state": "Haryana",
        "rto": "Rohtak",
        "model": "Maruti Celerio",
        "color": "White"
    },
    "HR01AB9012": {
        "owner_name": "Neetu Sharma",
        "vehicle_type": "Motorcycle",
        "state": "Haryana",
        "rto": "Gurgaon",
        "model": "Bajaj Dominar",
        "color": "Blue"
    },
    "WB06CD3456": {
        "owner_name": "Anirban Das",
        "vehicle_type": "Car",
        "state": "West Bengal",
        "rto": "Howrah",
        "model": "Tata Tigor",
        "color": "Silver"
    },
    "WB19EF7890": {
        "owner_name": "Moumita Sen",
        "vehicle_type": "Scooter",
        "state": "West Bengal",
        "rto": "Siliguri",
        "model": "Hero Pleasure",
        "color": "Pink"
    },
    "GJ01GH2345": {
        "owner_name": "Hardik Patel",
        "vehicle_type": "Car",
        "state": "Gujarat",
        "rto": "Ahmedabad",
        "model": "Hyundai Creta",
        "color": "White"
    },
    "GJ05IJ6789": {
        "owner_name": "Nisha Shah",
        "vehicle_type": "Motorcycle",
        "state": "Gujarat",
        "rto": "Surat",
        "model": "Honda Shine",
        "color": "Red"
    },
    "RJ14KL1234": {
        "owner_name": "Mahendra Singh",
        "vehicle_type": "Car",
        "state": "Rajasthan",
        "rto": "Jaipur",
        "model": "Mahindra XUV500",
        "color": "Black"
    },
    "RJ27MN5678": {
        "owner_name": "Kavita Rathore",
        "vehicle_type": "Scooter",
        "state": "Rajasthan",
        "rto": "Jodhpur",
        "model": "TVS Scooty Pep",
        "color": "Red"
    },
    "AP09OP9012": {
        "owner_name": "Venkatesh Reddy",
        "vehicle_type": "Car",
        "state": "Andhra Pradesh",
        "rto": "Visakhapatnam",
        "model": "Kia Sonet",
        "color": "Orange"
    },
    "AP39QR3456": {
        "owner_name": "Lakshmi Devi",
        "vehicle_type": "Motorcycle",
        "state": "Andhra Pradesh",
        "rto": "Anantapur",
        "model": "Yamaha FZ",
        "color": "Blue"
    },
    "TS07ST7890": {
        "owner_name": "Srinivas Rao",
        "vehicle_type": "Car",
        "state": "Telangana",
        "rto": "Hyderabad",
        "model": "Honda Amaze",
        "color": "Silver"
    },
    "TS09UV2345": {
        "owner_name": "Swapna Reddy",
        "vehicle_type": "Scooter",
        "state": "Telangana",
        "rto": "Warangal",
        "model": "Suzuki Burgman",
        "color": "White"
    },
    "PB03WX6789": {
        "owner_name": "Gurpreet Singh",
        "vehicle_type": "Car",
        "state": "Punjab",
        "rto": "Ludhiana",
        "model": "Maruti Swift",
        "color": "Red"
    },
    "PB10YZ1234": {
        "owner_name": "Simran Kaur",
        "vehicle_type": "Motorcycle",
        "state": "Punjab",
        "rto": "Amritsar",
        "model": "Royal Enfield Bullet",
        "color": "Black"
    },
    "OD02AB5678": {
        "owner_name": "Bijay Panda",
        "vehicle_type": "Car",
        "state": "Odisha",
        "rto": "Bhubaneswar",
        "model": "Tata Nexon",
        "color": "Blue"
    },
    "OD05CD9012": {
        "owner_name": "Anita Mohanty",
        "vehicle_type": "Scooter",
        "state": "Odisha",
        "rto": "Cuttack",
        "model": "Honda Activa",
        "color": "Grey"
    },
    "BR01EF3456": {
        "owner_name": "Ravi Kumar",
        "vehicle_type": "Car",
        "state": "Bihar",
        "rto": "Patna",
        "model": "Hyundai Grand i10",
        "color": "White"
    },
    "BR03GH7890": {
        "owner_name": "Sunita Devi",
        "vehicle_type": "Motorcycle",
        "state": "Bihar",
        "rto": "Gaya",
        "model": "Hero Passion Pro",
        "color": "Black"
    },
}


def normalize_plate(plate_number: str) -> str:
    """
    Normalize a plate number for lookup.
    
    Args:
        plate_number: Raw plate number (may have spaces, lowercase)
    
    Returns:
        Normalized plate number (uppercase, no spaces)
    """
    if not plate_number:
        return ""
    
    # Convert to uppercase and remove all spaces
    normalized = plate_number.upper().strip()
    normalized = re.sub(r'\s+', '', normalized)
    
    return normalized


def lookup_vehicle(plate_number: str) -> Dict:
    """
    Look up a vehicle in the mock registration database.
    
    Args:
        plate_number: Vehicle registration number (e.g., "TN10AB1234" or "TN 10 AB 1234")
    
    Returns:
        Dictionary with registration details if found, or unregistered status
        
    Examples:
        >>> lookup_vehicle("TN10AB1234")
        {
            "registered": True,
            "owner_name": "Ravi Kumar",
            "vehicle_type": "Car",
            "state": "Tamil Nadu",
            "rto": "Chennai South",
            "model": "Maruti Swift",
            "color": "White"
        }
        
        >>> lookup_vehicle("XX99ZZ9999")
        {
            "registered": False,
            "owner_name": None,
            "vehicle_type": None,
            "state": None,
            "rto": None,
            "model": None,
            "color": None
        }
    """
    # Normalize the plate number
    normalized_plate = normalize_plate(plate_number)
    
    # Look up in registry
    if normalized_plate in MOCK_REGISTRY:
        vehicle_info = MOCK_REGISTRY[normalized_plate].copy()
        vehicle_info["registered"] = True
        vehicle_info["plate_number"] = normalized_plate
        return vehicle_info
    else:
        return {
            "registered": False,
            "plate_number": normalized_plate,
            "owner_name": None,
            "vehicle_type": None,
            "state": None,
            "rto": None,
            "model": None,
            "color": None
        }


def is_registered_plate(plate_number: str) -> bool:
    """
    Check if a plate number is registered.
    
    Args:
        plate_number: Vehicle registration number
    
    Returns:
        True if registered, False otherwise
        
    Examples:
        >>> is_registered_plate("TN10AB1234")
        True
        
        >>> is_registered_plate("XX99ZZ9999")
        False
    """
    normalized_plate = normalize_plate(plate_number)
    return normalized_plate in MOCK_REGISTRY


def get_registry_stats() -> Dict:
    """
    Get statistics about the mock registry.
    
    Returns:
        Dictionary with registry statistics
    """
    total_vehicles = len(MOCK_REGISTRY)
    
    # Count by vehicle type
    vehicle_types = {}
    states = {}
    
    for plate, info in MOCK_REGISTRY.items():
        v_type = info.get("vehicle_type", "Unknown")
        state = info.get("state", "Unknown")
        
        vehicle_types[v_type] = vehicle_types.get(v_type, 0) + 1
        states[state] = states.get(state, 0) + 1
    
    return {
        "total_vehicles": total_vehicles,
        "vehicle_types": vehicle_types,
        "states": states
    }


# Example usage and testing
if __name__ == "__main__":
    print("=" * 60)
    print("RoadVision Mock Vehicle Registration Database")
    print("=" * 60)
    
    # Test lookups
    test_plates = [
        "TN10AB1234",
        "TN 10 AB 1234",  # With spaces
        "ka05mn7788",      # Lowercase
        "DL01AA4321",
        "XX99ZZ9999"       # Not registered
    ]
    
    print("\nTest Lookups:")
    print("-" * 60)
    for plate in test_plates:
        result = lookup_vehicle(plate)
        print(f"\nPlate: {plate}")
        print(f"  Registered: {result['registered']}")
        if result['registered']:
            print(f"  Owner: {result['owner_name']}")
            print(f"  Type: {result['vehicle_type']}")
            print(f"  State: {result['state']}")
            print(f"  Model: {result['model']}")
    
    # Registry stats
    print("\n" + "=" * 60)
    print("Registry Statistics:")
    print("-" * 60)
    stats = get_registry_stats()
    print(f"Total Vehicles: {stats['total_vehicles']}")
    print(f"\nBy Vehicle Type:")
    for v_type, count in stats['vehicle_types'].items():
        print(f"  {v_type}: {count}")
    print(f"\nBy State:")
    for state, count in stats['states'].items():
        print(f"  {state}: {count}")
    print("=" * 60)
