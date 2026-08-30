import pulp

def optimize_order_quantity(predicted_demand, current_inventory=0, capacity=5000,
                            ordering_cost=5.0, holding_cost=1.0, stockout_cost=10.0):
    # Single-period linear inventory decision.
    x = pulp.LpVariable("order_quantity", lowBound=0, upBound=capacity)
    ending_inventory = pulp.LpVariable("ending_inventory", lowBound=0)
    shortage = pulp.LpVariable("shortage", lowBound=0)

    model = pulp.LpProblem("Inventory_Optimization", pulp.LpMinimize)

    # Inventory balance: current stock + order + shortage = demand + ending stock
    model += current_inventory + x + shortage == predicted_demand + ending_inventory
    model += (ordering_cost * x +
              holding_cost * ending_inventory +
              stockout_cost * shortage)

    model.solve(pulp.PULP_CBC_CMD(msg=False))

    return {
        "status": pulp.LpStatus[model.status],
        "order_quantity": x.value(),
        "ending_inventory": ending_inventory.value(),
        "shortage": shortage.value(),
        "objective_value": pulp.value(model.objective),
    }
