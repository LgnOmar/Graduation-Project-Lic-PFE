import 'package:flutter/material.dart';
import 'package:jibjob/screens/ListePrix.dart';
import 'package:jibjob/screens/Pro/Profilepro.dart';
import 'package:image_picker/image_picker.dart';
import 'dart:io';
import 'package:jibjob/Pro.dart';


class Mesprix extends StatefulWidget {
  
  int Pro_Position ;
  
  Mesprix({
    required this.Pro_Position,
  }) ;

  @override
  _MesprixState createState() => _MesprixState();
}

class _MesprixState extends State<Mesprix> {

  final TextEditingController descriptionController = TextEditingController();
  final TextEditingController priceController = TextEditingController();
  final TextEditingController unitController = TextEditingController() ;



  void addTarif() {
    if (descriptionController.text.isNotEmpty &&
        priceController.text.isNotEmpty &&
        unitController.text.isNotEmpty) {

        setState(() {
          Liste_Pros[widget.Pro_Position].tarifs.add({
          "title": descriptionController.text,
          "price": "${priceController.text}DA/${unitController.text}"
        });

        Pros_Offers.add(Listeprix(
          rate: "4.5",
          name: Liste_Pros[widget.Pro_Position].name,
          address: Liste_Pros[widget.Pro_Position].address,
          image: Liste_Pros[widget.Pro_Position].image,
          tarifs: Liste_Pros[widget.Pro_Position].tarifs,
        ));


        });
        descriptionController.clear();
        priceController.clear();
        unitController.clear();
      
    }
  }

@override
  Widget build(BuildContext context) {

    Pro pro = Liste_Pros[widget.Pro_Position] ;


    return Scaffold(
      appBar: AppBar(
        backgroundColor: Colors.white,
        foregroundColor: Color(0xFF130160),
        elevation: 0,
        leading: IconButton(
      icon: Icon(Icons.arrow_back),
      onPressed: () {
        Navigator.push(
          context,
          MaterialPageRoute(
            builder: (_) => Profilepro(Pro_Position : widget.Pro_Position),
          ),
        );
      },
    ),
        title: Text("Mes Prix", style: TextStyle(color: Color(0xFF130160),)),
        centerTitle: true,
      ),

      body: SafeArea(
        child: SingleChildScrollView(
          padding: EdgeInsets.symmetric(horizontal: 24, vertical: 16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [

              Text(
              "En indiquant vos tarifs de manière transparente, vous inspirez confiance et vous augmentez votre visibilité",
              style: TextStyle(fontSize: 13, color: Colors.black87),
            ),
            SizedBox(height: 16),

            TextField(
              controller: descriptionController,
              decoration: InputDecoration(
                hintText: "ex: Pose carrelage",
                filled: true,
                fillColor: Colors.grey[200],
                border: OutlineInputBorder(borderSide: BorderSide.none),
              ),
            ),
             SizedBox(height: 18),
            Row(
              children: [
                Expanded(
                  child: TextField(
                    controller: priceController,
                    keyboardType: TextInputType.number,
                    decoration: InputDecoration(
                      hintText: "Prix",
                      filled: true,
                      fillColor: Colors.grey[200],
                      border: OutlineInputBorder(borderSide: BorderSide.none),
                    ),
                  ),
                ),
                SizedBox(width: 8),
                Container(
                  width: 60,
                  height: 48,
                  alignment: Alignment.center,
                  decoration: BoxDecoration(
                    color: Colors.grey[200],
                    borderRadius: BorderRadius.circular(4),
                  ),
                  child: Text("DA", style: TextStyle(fontWeight: FontWeight.bold)),
                ),
                SizedBox(width: 8),
                Expanded(
                  child: TextField(
                    controller: unitController,
                    decoration: InputDecoration(
                      hintText: "Unité, m², km...",
                      filled: true,
                      fillColor: Colors.grey[200],
                      border: OutlineInputBorder(borderSide: BorderSide.none),
                    ),
                  ),
                ),

                SizedBox(width: 8),

                ElevatedButton(
                  onPressed: addTarif ,
                  child: Text("Ajouter" , style: TextStyle(fontSize: 13 , color: Colors.white)),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.green ,
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(4),
                    ),
                    padding: EdgeInsets.symmetric(horizontal: 16, vertical: 14),
                  ),
                ),
              ],
            ),
            SizedBox(height: 20),

            Text("Exemples", style: TextStyle(fontWeight: FontWeight.bold)),
SizedBox(height: 10),
SizedBox(
  height: 400, // Define fixed height for the list inside the scroll view
  child: ListView.builder(
    itemCount: Liste_Pros[widget.Pro_Position].tarifs.length,
    itemBuilder: (context, index) {
      final item = Liste_Pros[widget.Pro_Position].tarifs[index];
      return Container(
        margin: EdgeInsets.only(bottom: 8),
        padding: EdgeInsets.symmetric(horizontal: 12, vertical: 10),
        decoration: BoxDecoration(
          color: Color(0xFFE7DFFF),
          borderRadius: BorderRadius.circular(8),
        ),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Text(item["title"]!, style: TextStyle(fontWeight: FontWeight.w600)),
            Text(item["price"]!,
                style: TextStyle(color: Color(0xFF130160), fontWeight: FontWeight.w600)),
          ],
        ),
      );
    },
  ),
),

SizedBox(height: 20,) ,

SizedBox(
              width: double.infinity,
              child: ElevatedButton(
                onPressed: () {
                  for (int i = 0; i < widget.Pro_Position + 1 ; i++) {
                    print("\nName :${Liste_Pros[i].name}\n");
                    print(Liste_Pros[i].tarifs);
                  }
                  Navigator.push(context, MaterialPageRoute(builder: (_) => Profilepro(Pro_Position : widget.Pro_Position)));
                },
                child: Text("Sauvegarder", style: TextStyle(fontSize: 18 , color : Colors.white )),
                style: ElevatedButton.styleFrom(
                    backgroundColor:  Color(0xFF130160) ,
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(15),
                    ),
                    padding: EdgeInsets.symmetric(horizontal: 16, vertical: 14),
                  ),
              ),
            ),




            ],
          ),
        ),
      ),
    );}
    
  }