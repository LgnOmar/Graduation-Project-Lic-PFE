import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'dart:io';
import 'package:jibjob/Person.dart';
import 'package:jibjob/Pro.dart';
import 'package:jibjob/screens/Client/NewOrder.dart';
import 'package:jibjob/screens/Client/Profileclient.dart';


class MakeOrderClient extends StatefulWidget {

  int Client_Position ;

  MakeOrderClient({
    required this.Client_Position,
  }) ;


  @override
  _MakeOrderClientState createState() => _MakeOrderClientState();
}



class _MakeOrderClientState extends State<MakeOrderClient> {
  String selectedFilter = 'Tous';

  // Add this variable to track the selected nav index
  int selectedNavIndex = 1;



@override
  Widget build(BuildContext context) {
    return Scaffold(
      bottomNavigationBar: Stack(
  alignment: Alignment.center,
  children: [
    Container(
      height: 95,
      decoration: BoxDecoration(
        color: Colors.white,
        boxShadow: [
          BoxShadow(
            color: Colors.black12,
            blurRadius: 8,
            offset: Offset(0, -2),
          ),
        ],
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceAround,
        children: [
          _NavBarItem(
            icon: Icons.groups,
            label: "Pros",
            selected: selectedNavIndex == 0,
            onTap: () => setState(() => selectedNavIndex = 0),
          ),
          _NavBarItem(
            icon: Icons.campaign,
            label: "Demandes",
            selected: selectedNavIndex == 1,
            onTap: () => setState(() => selectedNavIndex = 1),
          ),
          SizedBox(width: 56), // Space for the FAB
          _NavBarItem(
            icon: Icons.message,
            label: "Messages",
            selected: selectedNavIndex == 2,
            onTap: () => setState(() => selectedNavIndex = 2),
          ),
          _NavBarItem(
            icon: Icons.person,
            label: "Compte",
            selected: selectedNavIndex == 3,
            onTap: () => setState(
              () {
                selectedNavIndex = 3;
                Navigator.push(context, MaterialPageRoute(builder: (_) => Profileclient(Client_Position : widget.Client_Position,)));
              }),
          ),
        ],
      ),
    ),
    Positioned(
  bottom: 20,
  child: ClipOval(
    child: Container(
      width: 60,
      height: 60,
      child: FloatingActionButton(
        onPressed: () {
          Navigator.push(
            context,
            MaterialPageRoute(
              builder: (context) => NewOrder(),
            ),
          );
        },
        backgroundColor: Color(0xFF20004E),
        elevation: 4,
        child: Icon(Icons.add, size: 40),
      ),
    ),
  ),
)

,
  ],
),



      
      appBar: PreferredSize(
        preferredSize: Size.fromHeight(120),
        child: AppBar(
          backgroundColor: Colors.white,
          elevation: 0,
          automaticallyImplyLeading: false,
          flexibleSpace: SafeArea(
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16.0, vertical: 12),
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Text(
                    'Professionels',
                    style: TextStyle(
                      color: Colors.deepPurple,
                      fontSize: 20,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  SizedBox(height: 12),
                  Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      FilterButton(
                        label: 'Nouveaux',
                        isSelected: selectedFilter == 'Nouveaux',
                        onTap: () => setState(() => selectedFilter = 'Nouveaux'),
                      ),
                      SizedBox(width: 8),
                      FilterButton(
                        label: 'Bien Notés',
                        isSelected: selectedFilter == 'Bien Notés',
                        onTap: () => setState(() => selectedFilter = 'Bien Notés'),
                      ),
                      SizedBox(width: 8),
                      FilterButton(
                        label: 'Mal Notés',
                        isSelected: selectedFilter == 'Mal Notés',
                        onTap: () => setState(() => selectedFilter = 'Mal Notés'),
                      ),
                      SizedBox(width: 8),
                      CircleAvatar(
                        radius: 18,
                        backgroundColor: Colors.deepPurpleAccent,
                        child: Icon(Icons.filter_list, color: Colors.white, size: 20),
                      ),
                    ],
                  ),
                ],
              ),
            ),
          ),
        ),
      ),

      body: SafeArea(
        child: SingleChildScrollView(
          padding: EdgeInsets.symmetric(horizontal: 24, vertical: 16),
          child :
          Container(
            color: Colors.white,
            height: 590 ,
          
          child :ListView.builder(
    itemCount: Liste_Pros.length,
    itemBuilder: (context, index) {
      final item = Liste_Pros[index];

      return Container(
  padding: EdgeInsets.all(16),
  decoration: BoxDecoration(
    color: Colors.white,
    borderRadius: BorderRadius.circular(20),
    boxShadow: [
      BoxShadow(
        color: Colors.black12,
        blurRadius: 8,
        offset: Offset(0, 2),
      ),
    ],
  ),
  child: Column(
    crossAxisAlignment: CrossAxisAlignment.start,
    children: [
      Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Stack(
            children: [
              CircleAvatar(
                      radius: 32,
                      backgroundImage: item.image == null
                          ? AssetImage('assets/_Unkown.png')
                          : FileImage(File(item.image!)),
                    ),

              Positioned(
                bottom: 0,
                right: 0,
                child: Container(
                  width: 16,
                  height: 16,
                  decoration: BoxDecoration(
                    color: Colors.green,
                    shape: BoxShape.circle,
                    border: Border.all(color: Colors.white, width: 2),
                  ),
                ),
              ),
            ],
          ),
          SizedBox(width: 16),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  item.name == null ?  "null" : item.name!,
                  style: TextStyle(
                    fontWeight: FontWeight.bold,
                    fontSize: 22,
                  ),
                ),
                SizedBox(height: 4),
                Row(
                  children: [
                    Icon(Icons.star, color: Colors.amber, size: 18),
                    SizedBox(width: 2),
                    Text(
                      "4.5",
                      style: TextStyle(fontWeight: FontWeight.bold),
                    ),
                    SizedBox(width: 4),
                    Text(
                      "(150 avis)",
                      style: TextStyle(color: Colors.grey[700], fontSize: 13),
                    ),
                  ],
                ),
                SizedBox(height: 4),
                Container(
                  padding: EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                  decoration: BoxDecoration(
                    color: Color(0xFFE7DFFF),
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: Text(
                    item.address == null ? "null" : item.address!,
                    style: TextStyle(fontSize: 13, color: Colors.black87),
                  ),
                ),
              ],
            ),
          ),
          SizedBox(width: 10),
          Column(
            children: [
              ElevatedButton.icon(
                onPressed: () {},
                icon: Icon(Icons.phone , color: Colors.white, size: 18),
                label: Text("Whatsapp"),
                style: ElevatedButton.styleFrom(
                  backgroundColor: Color(0xFF25D366),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(10),
                  ),
                  padding: EdgeInsets.symmetric(horizontal: 8, vertical: 6),
                  textStyle: TextStyle(fontSize: 13),
                  elevation: 0,
                ),
              ),
              ElevatedButton.icon(
                onPressed: () {},
                icon: Icon(Icons.phone, color: Colors.white, size: 18),
                label: Text("Appeler    "),
                style: ElevatedButton.styleFrom(
                  backgroundColor: Color(0xFF27AE60),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(10),
                  ),
                  padding: EdgeInsets.symmetric(horizontal: 8, vertical: 6),
                  textStyle: TextStyle(fontSize: 13),
                  elevation: 0,
                ),
              ),
            ],
          ),
        ],
      ),
      SizedBox(height: 16),
      Text(
        "A votre disposition",
        style: TextStyle(fontSize: 15),
      ),
      SizedBox(height: 12),
      Text(
        "Prix",
        style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16),
      ),
      SizedBox(height: 8),
      Column(
        children: [
          for (var tarif in item.tarifs)
            _priceRow(tarif["title"]!, tarif["price"]!),
        ],
      ),
    ],)
  );
    },
  )),
        ),
      ),
    );}
    
  }


  class FilterButton extends StatelessWidget {
  final String label;
  final bool isSelected;
  final VoidCallback? onTap; // Add this

  const FilterButton({
    required this.label,
    this.isSelected = false,
    this.onTap, // Add this
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap, // Add this
      child: Container(
        padding: EdgeInsets.symmetric(horizontal: 16, vertical: 8),
        decoration: BoxDecoration(
          color: isSelected ? Colors.deepPurpleAccent : Colors.grey[300],
          borderRadius: BorderRadius.circular(20),
        ),
        child: Text(
          label,
          style: TextStyle(
            color: isSelected ? Colors.white : Colors.black87,
            fontWeight: FontWeight.w500,
          ),
        ),
      ),
    );
  }
}


// Helper widget for price row
Widget _priceRow(String title, String price) {
  return Container(
    margin: EdgeInsets.only(bottom: 6),
    padding: EdgeInsets.symmetric(horizontal: 12, vertical: 10),
    decoration: BoxDecoration(
      color: Color(0xFFE7DFFF),
      borderRadius: BorderRadius.circular(8),
    ),
    child: Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        Text(
          title,
          style: TextStyle(
            fontWeight: FontWeight.w600,
            color: Color(0xFF130160),
          ),
        ),
        Text(
          price,
          style: TextStyle(
            color: Color(0xFF7B61FF),
            fontWeight: FontWeight.bold,
          ),
        ),
      ],
    ),
  );
}

class _NavBarItem extends StatelessWidget {
  final IconData icon;
  final String label;
  final bool selected;
  final VoidCallback? onTap;

  const _NavBarItem({
    required this.icon,
    required this.label,
    this.selected = false,
    this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(
            icon,
            color: selected ? Color(0xFF20004E) : Colors.deepPurple.shade100,
          ),
          SizedBox(height: 2),
          Text(
            label,
            style: TextStyle(
              fontSize: 12,
              color: selected ? Color(0xFF20004E) : Colors.deepPurple.shade100,
              fontWeight: selected ? FontWeight.bold : FontWeight.normal,
            ),
          ),
        ],
      ),
    );
  }
}