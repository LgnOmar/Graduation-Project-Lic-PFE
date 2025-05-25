import 'package:flutter/material.dart';
import 'package:jibjob/screens/Offre.dart';
import 'package:jibjob/screens/Pro/Profilepro.dart';

class ListeChoix extends StatefulWidget {

    int Pro_Position ;

  ListeChoix({
    required this.Pro_Position,
  }) ;


  @override
  _ListeChoixState createState() => _ListeChoixState();
}

class _ListeChoixState extends State<ListeChoix> {

  int selectedNavIndex = 1;
  

  @override
  Widget build(BuildContext context) {
    final String avatarUrl =
        'https://randomuser.me/api/portraits/men/1.jpg';

    return Scaffold(

                  bottomNavigationBar : Stack(
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
            onTap: () => setState(() {
              selectedNavIndex = 1 ;
              }),
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
                Navigator.push(context, MaterialPageRoute(builder: (_) => Profilepro(Pro_Position : widget.Pro_Position,)));
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
        onPressed: () {},
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

      appBar: AppBar(
        backgroundColor: Colors.white,
        elevation: 0,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back, color: Colors.black),
          onPressed: () => Navigator.of(context).pop(),
        ),
        title: const Text(
          'Demandes',
          style: TextStyle(
            color: Colors.black,
            fontWeight: FontWeight.w600,
            fontSize: 24,
          ),
        ),
        centerTitle: true,
        actions: [
          Padding(
            padding: const EdgeInsets.only(right: 16.0),
            child: CircleAvatar(
              backgroundColor: Color(0xFF6C63FF),
              child: Icon(Icons.filter_list, color: Colors.white),
            ),
          ),
        ],
      ),
      backgroundColor: Colors.white,
      body: Column(
        children: [
          // Filter Row - Make it horizontally scrollable
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 16.0, vertical: 8),
            child: SingleChildScrollView(
              scrollDirection: Axis.horizontal,
              child: Row(
                children: [
                  ElevatedButton.icon(
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Color(0xFF6C63FF),
                      foregroundColor: Colors.white,
                      shape: StadiumBorder(),
                      elevation: 0,
                    ),
                    icon: Icon(Icons.thumb_up_alt_outlined, size: 18),
                    label: Text('Recommandee'),
                    onPressed: () {},
                  ),
                  SizedBox(width: 8),
                  FilterChip(
                    label: Text('Tout'),
                    avatar: Icon(Icons.public, size: 18, color: Colors.grey),
                    selected: false,
                    onSelected: (_) {},
                    backgroundColor: Color(0xFFF2F2F2),
                    shape: StadiumBorder(),
                  ),
                  SizedBox(width: 8),
                  FilterChip(
                    label: Text('Avis'),
                    avatar: Icon(Icons.sync_alt, size: 18, color: Colors.grey),
                    selected: false,
                    onSelected: (_) {},
                    backgroundColor: Color(0xFFF2F2F2),
                    shape: StadiumBorder(),
                  ),
                  SizedBox(width: 8),
                  FilterChip(
                    label: Text('Sans'),
                    selected: false,
                    onSelected: (_) {},
                    backgroundColor: Color(0xFFF2F2F2),
                    shape: StadiumBorder(),
                  ),
                ],
              ),
            ),
          ),
          Divider(height: 1),
          // Demande Card
          Expanded(
            child : Container(
              padding: const EdgeInsets.all(16.0),

            child: ListView.builder(

              
                itemCount: Liste_Offres.length,
                itemBuilder: (context, index) {
                final item = Liste_Offres[index];

                return Card(
                  elevation: 0,
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(16),
                  ),

                  color: Colors.white,
                  child: Padding(
                    padding: const EdgeInsets.all(16.0),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        // Status Row
                        Wrap(
                          spacing: 8,
                          crossAxisAlignment: WrapCrossAlignment.center,
                          children: [
                            Text(
                              'Ouverte',
                              style: TextStyle(
                                color: Color(0xFF2ECC40),
                                fontWeight: FontWeight.bold,
                              ),
                            ),
                            Text(
                              '• Il y a 2 heures',
                              style: TextStyle(color: Colors.grey, fontSize: 13),
                            ),
                            Text(
                              '• ${item.address ?? 'null'}',
                              style: TextStyle(color: Colors.grey, fontSize: 13),
                            ),
                            Text(
                              '• 23km',
                              style: TextStyle(color: Colors.grey, fontSize: 13),
                            ),
                            Text(
                              '• 21 offres',
                              style: TextStyle(color: Colors.grey, fontSize: 13),
                            ),
                          ],
                        ),
                        SizedBox(height: 8),
                        // Points badge
                        Container(
                          padding: EdgeInsets.symmetric(horizontal: 8, vertical: 2),
                          decoration: BoxDecoration(
                            color: Color(0xFFFFD600),
                            borderRadius: BorderRadius.circular(8),
                          ),
                          child: Text(
                            '20p',
                            style: TextStyle(
                              color: Colors.black,
                              fontWeight: FontWeight.bold,
                              fontSize: 13,
                            ),
                          ),
                        ),
                        SizedBox(height: 8),
                        // Title
                        Text(
                          item.demande ?? 'null',
                          style: TextStyle(
                            color: Color(0xFF2D1E46),
                            fontWeight: FontWeight.bold,
                            fontSize: 20,
                          ),
                        ),
                        SizedBox(height: 12),
                        // Avatars Row
                        SizedBox(
                          height: 36,
                          child: ListView.separated(
                            scrollDirection: Axis.horizontal,
                            itemCount: 14,
                            separatorBuilder: (_, __) => SizedBox(width: 4),
                            itemBuilder: (context, index) => CircleAvatar(
                              radius: 18,
                              backgroundImage: NetworkImage(avatarUrl),
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                ) ;
  })
            ),
          ),
        ],
      ),
    );
  }
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