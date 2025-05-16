import 'package:flutter/material.dart';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: ProfessionalPage(),
      debugShowCheckedModeBanner: false,
    );
  }
}

class ProfessionalPage extends StatelessWidget {
  final Color darkPurple = Color(0xFF20004E);
  final Color green = Color(0xFF25D366); // WhatsApp green

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white,
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: SingleChildScrollView(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // Header with filters
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Text(
                    "Professionnels",
                    style: TextStyle(
                      fontSize: 24,
                      fontWeight: FontWeight.bold,
                      color: darkPurple,
                    ),
                  ),
                  IconButton(
                    icon: Icon(Icons.filter_alt, color: darkPurple),
                    onPressed: () {
                      // Add filter logic here
                    },
                  ),
                ],
              ),
              SizedBox(height: 16),

              // Profile Section for Multiple Listings
              ListView.builder(
                shrinkWrap: true,
                itemCount: 5, // Number of profiles
                itemBuilder: (context, index) {
                  return ProfessionalCard();
                },
              ),

              // Bottom Navigation Bar with "+"
              SizedBox(height: 32),
              BottomNavigationBar(
                selectedItemColor: darkPurple,
                unselectedItemColor: Colors.grey,
                items: const [
                  BottomNavigationBarItem(icon: Icon(Icons.person), label: 'Pros'),
                  BottomNavigationBarItem(icon: Icon(Icons.list), label: 'Demandes'),
                  BottomNavigationBarItem(
                    icon: Icon(Icons.add_circle, size: 40),
                    label: '',
                  ),
                  BottomNavigationBarItem(icon: Icon(Icons.message), label: 'Messages'),
                  BottomNavigationBarItem(icon: Icon(Icons.account_circle), label: 'Compte'),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class ProfessionalCard extends StatelessWidget {
  final Color darkPurple = Color(0xFF20004E);
  final Color green = Color(0xFF25D366);

  @override
  Widget build(BuildContext context) {
    return Card(
      margin: EdgeInsets.only(bottom: 16),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      elevation: 4,
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            // Profile and Rating
            Row(
              children: [
                CircleAvatar(
                  radius: 40,
                  //backgroundImage: AssetImage('assets/profile_image.png'), // Replace with actual image
                ),
                SizedBox(width: 16),
                Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'Omar',
                      style: TextStyle(fontSize: 22, fontWeight: FontWeight.bold, color: darkPurple),
                    ),
                    SizedBox(height: 4),
                    Row(
                      children: [
                        Text(
                          '4.8',
                          style: TextStyle(fontSize: 16, fontWeight: FontWeight.w600, color: Colors.black),
                        ),
                        Icon(Icons.star, color: Colors.yellow, size: 18),
                        SizedBox(width: 8),
                        Text('(150 avis)', style: TextStyle(fontSize: 14, color: Colors.grey)),
                      ],
                    ),
                    SizedBox(height: 8),
                    Row(
                      children: [
                        Icon(Icons.location_on, color: darkPurple, size: 20),
                        SizedBox(width: 8),
                        Text(
                          'Oued Smar, Alger',
                          style: TextStyle(fontSize: 14, color: Colors.black54),
                        ),
                      ],
                    ),
                  ],
                ),
              ],
            ),
            SizedBox(height: 32),
            // Contact Buttons (Whatsapp, Call)
            Row(
              children: [
                ElevatedButton(
                  onPressed: () {},
                  style: ElevatedButton.styleFrom(
                    backgroundColor: green,
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(12),
                    ),
                  ),
                  child: Row(
                    children: [
                      Icon(Icons.message, color: Colors.white),
                      SizedBox(width: 8),
                      Text("Whatsapp", style: TextStyle(color: Colors.white)),
                    ],
                  ),
                ),
                SizedBox(width: 16),
                ElevatedButton(
                  onPressed: () {},
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.green[600],
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(12),
                    ),
                  ),
                  child: Row(
                    children: [
                      Icon(Icons.call, color: Colors.white),
                      SizedBox(width: 8),
                      Text("Appeler", style: TextStyle(color: Colors.white)),
                    ],
                  ),
                ),
              ],
            ),
            SizedBox(height: 32),

            // Service List
            Text(
              'A votre disposition',
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold, color: darkPurple),
            ),
            SizedBox(height: 16),
            // Service 1
            ServiceCard(service: 'Climatisateur 12000', price: '5000DA'),
            ServiceCard(service: 'Climatisateur 18000', price: '6000DA'),
            ServiceCard(service: 'Demonte Clim', price: '3000DA'),
            ServiceCard(service: 'Devis Deplacement', price: '2500DA'),
          ],
        ),
      ),
    );
  }
}

class ServiceCard extends StatelessWidget {
  final String service;
  final String price;

  const ServiceCard({required this.service, required this.price});

  @override
  Widget build(BuildContext context) {
    return Card(
      margin: EdgeInsets.only(bottom: 16),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      elevation: 4,
      child: ListTile(
        title: Text(
          service,
          style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
        ),
        trailing: Text(
          price,
          style: TextStyle(fontSize: 16, fontWeight: FontWeight.w600),
        ),
      ),
    );
  }
}
